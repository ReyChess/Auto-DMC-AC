#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#if !defined(_WIN32) || defined(__CYGWIN__)
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static constexpr int STABLE_K_MAX = 12;
static constexpr double STABLE_BETA = 2.0;
static constexpr double STABLE_BETA_NEG = 15.0;
static constexpr double STABLE_TIE_EPS = 0.005;
static constexpr int STABLE_K_NEG_MAX = 2;
static constexpr double DEFAULT_LAMBDA_PRIOR = 0.15;
static constexpr double M4_NEG_VETO_MIN = 0.10;
static constexpr double M4_NEG_VETO_DIFF = 0.05;
static constexpr double NUM_EPS = 1e-12;
static constexpr double NEG_INF = -1e300;

struct Rule {
    std::vector<int> antecedent;
    int class_label{};
    bool negated{};
    double support_antecedent{};
    double support_class{};
    double support_rule{};
    double netconf{};
};

struct RuleStatistics {
    int total_rules{};
    int positive_rules{};
    int negative_rules{};
};

struct CoveredRuleLog {
    std::vector<int> antecedent;
    double netconf{};
};

struct ClassEvidenceLog {
    int class_label{};
    double positive_score{NEG_INF};
    double negative_score{};
    std::vector<CoveredRuleLog> positive_rules;
    std::vector<CoveredRuleLog> negative_rules;
};

struct InstanceLog {
    int instance_id{};
    std::vector<int> items;
    int true_class{};
    int predicted_class{};
    int correct{};
    int default_used{};
    int partial_used{};
    int exact_positive_rules{};
    int partial_positive_rules{};
    int positive_cover_total{};
    double best_positive_score{NEG_INF};
    double second_positive_score{NEG_INF};
    double positive_margin{};
    double mu0_positive{};
    int near_tie{};
    int tie_size{};
    int negative_evaluated{};
    int negative_covered{};
    int negative_rules_activated{};
    int base_class{-1};
    double base_positive_score{NEG_INF};
    double base_negative_score{};
    int alternative_class{-1};
    double alternative_positive_score{NEG_INF};
    double alternative_negative_score{};
    int veto_condition_met{};
    int veto_changed_prediction{};
    int veto_change_correct{};
    int veto_change_incorrect{};
    std::vector<ClassEvidenceLog> class_evidence;
};

struct EvalResult {
    double accuracy{};
    double macro_f1{};
    int correct{};
    int total{};
    int default_count{};
    int partial_used_count{};
    long long exact_positive_rule_count{};
    long long partial_positive_rule_count{};
    long long positive_cover_count{};
    int near_tie_count{};
    long long total_tie_classes{};
    int negative_evaluated_count{};
    int negative_covered_count{};
    long long negative_rule_activation_count{};
    int veto_condition_count{};
    int veto_change_count{};
    int veto_correct_change_count{};
    int veto_incorrect_change_count{};
    double classification_seconds{};
    std::vector<int> truths;
    std::vector<int> predictions;
    std::vector<InstanceLog> instances;
};

struct Variant {
    std::string name;
    std::string support_mode;
    std::string evidence_mode;
    bool uses_negatives() const { return evidence_mode == "pn"; }
    bool uses_cns() const { return support_mode == "cns"; }
};

struct RunMetrics {
    Variant variant;
    double lambda_prior{DEFAULT_LAMBDA_PRIOR};
    int outer_fold{};
    int inner_fold{}; // 0 means final outer evaluation
    std::string phase; // inner or outer
    double sigma{};
    int training_instances{};
    int validation_instances{};
    RuleStatistics rule_stats;
    double mining_seconds{};
    EvalResult eval;
    double total_seconds() const { return mining_seconds + eval.classification_seconds; }
};

struct VariantScore {
    Variant variant;
    double sigma{};
    double lambda_prior{DEFAULT_LAMBDA_PRIOR};
    double mean_macro_f1{};
    double sd_macro_f1{};
    double mean_accuracy{};
    double mean_rules{};
    double mean_positive_rules{};
    double mean_negative_rules{};
    double mean_mining_seconds{};
    double mean_classification_seconds{};
    double mean_total_seconds{};
};

using Row = std::vector<int>;
using Dataset = std::vector<Row>;
using RulesByClass = std::map<int, std::vector<Rule>>;

struct LogWriters {
    std::ofstream instance;
    std::ofstream run_summary;
    std::ofstream selection;
    std::ofstream outer_summary;
    std::ofstream pn_guard;
    std::ofstream cns_guard;
};

static std::string quote(const fs::path& p) {
#if defined(_WIN32) && !defined(__CYGWIN__)
    return "\"" + p.string() + "\"";
#else
    std::string out = "'";
    for (char ch : p.string()) out += (ch == '\'' ? "'\\''" : std::string(1, ch));
    out += "'";
    return out;
#endif
}

static Dataset read_dataset(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open dataset: " + path.string());
    Dataset rows;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) continue;
        std::istringstream iss(line);
        Row row;
        int x;
        while (iss >> x) row.push_back(x);
        if (row.size() < 2) throw std::runtime_error("Invalid transaction at " + path.string() + ":" + std::to_string(line_no));
        rows.push_back(std::move(row));
    }
    if (rows.empty()) throw std::runtime_error("Empty dataset: " + path.string());
    return rows;
}

static void write_dataset(const fs::path& path, const Dataset& rows) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write dataset: " + path.string());
    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i) out << ' ';
            out << row[i];
        }
        out << '\n';
    }
}

static std::map<int,int> class_counts(const Dataset& rows) {
    std::map<int,int> counts;
    for (const auto& row : rows) counts[row.back()]++;
    return counts;
}

static std::vector<Dataset> stratified_folds(const Dataset& rows, int k, unsigned seed) {
    if (k < 2) throw std::runtime_error("inner_folds must be >= 2");
    std::map<int, Dataset> grouped;
    for (const auto& row : rows) grouped[row.back()].push_back(row);
    std::vector<Dataset> folds(static_cast<size_t>(k));
    std::mt19937 rng(seed);
    for (auto& [cls, group] : grouped) {
        (void)cls;
        std::shuffle(group.begin(), group.end(), rng);
        for (size_t i = 0; i < group.size(); ++i) folds[i % static_cast<size_t>(k)].push_back(group[i]);
    }
    for (auto& fold : folds) std::shuffle(fold.begin(), fold.end(), rng);
    return folds;
}

static RulesByClass parse_rules(const fs::path& path, RuleStatistics* stats = nullptr) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open rules: " + path.string());
    RulesByClass result;
    RuleStatistics local;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) continue;
        std::istringstream iss(line);
        int n_items;
        if (!(iss >> n_items) || n_items < 1) throw std::runtime_error("Invalid rule at " + path.string() + ":" + std::to_string(line_no));
        std::vector<int> items(static_cast<size_t>(n_items));
        for (int& item : items) if (!(iss >> item)) throw std::runtime_error("Incomplete rule items at line " + std::to_string(line_no));
        Rule r;
        int raw_class = items.back();
        r.class_label = std::abs(raw_class);
        r.negated = raw_class < 0;
        r.antecedent.assign(items.begin(), items.end() - 1);
        if (!(iss >> r.support_antecedent >> r.support_class >> r.support_rule >> r.netconf))
            throw std::runtime_error("Incomplete rule metrics at line " + std::to_string(line_no));
        result[r.class_label].push_back(std::move(r));
        ++local.total_rules;
        if (raw_class < 0) ++local.negative_rules; else ++local.positive_rules;
    }
    for (auto& [cls, rules] : result) {
        (void)cls;
        std::stable_sort(rules.begin(), rules.end(), [](const Rule& a, const Rule& b) {
            if (a.antecedent.size() != b.antecedent.size()) return a.antecedent.size() > b.antecedent.size();
            return a.netconf > b.netconf;
        });
    }
    if (stats) *stats = local;
    return result;
}

static double macro_f1(const std::vector<int>& truth, const std::vector<int>& pred, const std::vector<int>& labels) {
    if (labels.empty()) return 0.0;
    double sum = 0.0;
    for (int c : labels) {
        int tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < truth.size(); ++i) {
            if (truth[i] == c && pred[i] == c) ++tp;
            else if (truth[i] != c && pred[i] == c) ++fp;
            else if (truth[i] == c && pred[i] != c) ++fn;
        }
        int denom = 2 * tp + fp + fn;
        sum += denom ? (2.0 * tp / static_cast<double>(denom)) : 0.0;
    }
    return sum / labels.size();
}

static double mean(const std::vector<double>& x) {
    return x.empty() ? 0.0 : std::accumulate(x.begin(), x.end(), 0.0) / x.size();
}

static double sample_sd(const std::vector<double>& x) {
    if (x.size() < 2) return 0.0;
    double m = mean(x), ss = 0.0;
    for (double v : x) ss += (v-m)*(v-m);
    return std::sqrt(ss / static_cast<double>(x.size()-1));
}

static double pos_score(const std::vector<const Rule*>& rules, double mu0) {
    if (rules.empty()) return NEG_INF;
    int K = std::min<int>(static_cast<int>(rules.size()), STABLE_K_MAX);
    double num = 0.0, den = 0.0;
    for (int j = 0; j < K; ++j) {
        double w = 1.0 / (j + 1.0);
        num += w * rules[static_cast<size_t>(j)]->netconf;
        den += w;
    }
    double avg = den > 0 ? num / den : 0.0;
    return (avg * K + STABLE_BETA * mu0) / (K + STABLE_BETA);
}

static double neg_score(const std::vector<const Rule*>& rules) {
    std::vector<double> strengths;
    for (const Rule* r : rules) if (-r->netconf > 0.0) strengths.push_back(-r->netconf);
    std::sort(strengths.begin(), strengths.end(), std::greater<double>());
    if (strengths.empty()) return 0.0;
    int used = std::min<int>(static_cast<int>(strengths.size()), STABLE_K_NEG_MAX);
    return strengths.front() * used / (used + STABLE_BETA_NEG);
}

static bool exact_cover(const Rule& r, const std::set<int>& instance) {
    return std::all_of(r.antecedent.begin(), r.antecedent.end(), [&](int item){ return instance.count(item) != 0; });
}

static bool one_missing_cover(const Rule& r, const std::set<int>& instance) {
    int missing = 0;
    for (int item : r.antecedent) {
        if (!instance.count(item) && ++missing > 1) return false;
    }
    return missing == 1;
}

static EvalResult classify(const RulesByClass& rules_by_class,
                           const Dataset& rows,
                           const std::map<int,int>& counts,
                           double lambda_prior,
                           bool collect_rule_evidence = false) {
    auto t0 = Clock::now();
    EvalResult out;
    std::vector<int> labels;
    int ntrain = 0;
    for (const auto& [c, n] : counts) { labels.push_back(c); ntrain += n; }
    if (labels.empty() || ntrain <= 0) throw std::runtime_error("No class counts available for classification");

    std::map<int,double> prior;
    for (const auto& [c, n] : counts) prior[c] = n / static_cast<double>(ntrain);
    int default_cls = labels.front();
    for (int c : labels) {
        if (prior[c] > prior[default_cls] + NUM_EPS ||
            (std::fabs(prior[c] - prior[default_cls]) <= NUM_EPS && c < default_cls)) default_cls = c;
    }

    int instance_id = 0;
    for (const auto& row : rows) {
        ++instance_id;
        InstanceLog il;
        il.instance_id = instance_id;
        il.true_class = row.back();
        if (collect_rule_evidence) il.items.assign(row.begin(), row.end() - 1);
        std::set<int> instance(row.begin(), row.end() - 1);
        out.truths.push_back(il.true_class);

        std::map<int, std::vector<const Rule*>> pos;
        for (int c : labels) {
            auto it = rules_by_class.find(c);
            if (it == rules_by_class.end()) continue;
            for (const Rule& r : it->second) {
                if (!r.negated && exact_cover(r, instance)) pos[c].push_back(&r);
            }
        }
        int exact_total = 0;
        for (const auto& [c, v] : pos) { (void)c; exact_total += static_cast<int>(v.size()); }
        il.exact_positive_rules = exact_total;
        out.exact_positive_rule_count += exact_total;

        if (exact_total == 0) {
            for (int c : labels) {
                auto it = rules_by_class.find(c);
                if (it == rules_by_class.end()) continue;
                for (const Rule& r : it->second) {
                    if (!r.negated && one_missing_cover(r, instance)) {
                        pos[c].push_back(&r);
                        ++il.partial_positive_rules;
                    }
                }
            }
            if (il.partial_positive_rules > 0) {
                il.partial_used = 1;
                ++out.partial_used_count;
                out.partial_positive_rule_count += il.partial_positive_rules;
            }
        }

        for (const auto& [c, v] : pos) { (void)c; il.positive_cover_total += static_cast<int>(v.size()); }
        out.positive_cover_count += il.positive_cover_total;

        int prediction = default_cls;
        std::map<int,double> score;
        std::map<int,std::vector<const Rule*>> neg;
        std::map<int,double> negative_scores;
        std::vector<double> all_pos;
        for (const auto& [c, v] : pos) { (void)c; for (const Rule* r : v) all_pos.push_back(r->netconf); }

        if (all_pos.empty()) {
            il.default_used = 1;
            ++out.default_count;
        } else {
            il.mu0_positive = std::accumulate(all_pos.begin(), all_pos.end(), 0.0) / all_pos.size();
            int best_pos_c = -1, best_base_c = -1;
            double best_pos = NEG_INF, best_base = NEG_INF;

            for (int c : labels) {
                score[c] = pos_score(pos[c], il.mu0_positive);
                if (score[c] <= -1e200) continue;
                if (best_pos_c < 0 || score[c] > best_pos + NUM_EPS ||
                    (std::fabs(score[c]-best_pos)<=NUM_EPS && c < best_pos_c)) {
                    best_pos = score[c]; best_pos_c = c;
                }
                double base = score[c] + lambda_prior * std::log(prior[c]);
                if (best_base_c < 0 || base > best_base + NUM_EPS ||
                    (std::fabs(base-best_base)<=NUM_EPS && c < best_base_c)) {
                    best_base = base; best_base_c = c;
                }
            }

            il.best_positive_score = best_pos;
            for (int c : labels) {
                if (c == best_pos_c || score[c] <= -1e200) continue;
                if (il.second_positive_score <= -1e200 || score[c] > il.second_positive_score) il.second_positive_score = score[c];
            }
            if (il.second_positive_score > -1e200) il.positive_margin = best_pos - il.second_positive_score;

            if (best_pos_c < 0) {
                il.default_used = 1;
                ++out.default_count;
            } else {
                std::vector<int> tied;
                for (int c : labels) if (score[c] > -1e200 && best_pos - score[c] <= STABLE_TIE_EPS) tied.push_back(c);
                il.tie_size = static_cast<int>(tied.size());
                out.total_tie_classes += il.tie_size;

                if (tied.size() <= 1) {
                    prediction = best_base_c;
                    il.base_class = best_base_c;
                    il.base_positive_score = score[best_base_c];
                } else {
                    il.near_tie = 1;
                    ++out.near_tie_count;
                    il.negative_evaluated = 1;
                    ++out.negative_evaluated_count;

                    // v5.4.2: the prior-adjusted positive base decision is
                    // global over every class with positive evidence.  The
                    // unadjusted near-tie set is only the region in which
                    // negative counter-evidence may be inspected; it must not
                    // restrict the positive argmax.
                    const int base_c = best_base_c;
                    il.base_class = base_c;
                    il.base_positive_score = score[base_c];

                    for (int c : tied) {
                        auto it = rules_by_class.find(c);
                        if (it == rules_by_class.end()) continue;
                        for (const Rule& r : it->second) {
                            if (r.negated && exact_cover(r, instance)) {
                                neg[c].push_back(&r);
                                ++il.negative_rules_activated;
                            }
                        }
                    }
                    if (il.negative_rules_activated > 0) {
                        il.negative_covered = 1;
                        ++out.negative_covered_count;
                    }
                    out.negative_rule_activation_count += il.negative_rules_activated;

                    int min_c = -1;
                    double min_neg = std::numeric_limits<double>::infinity();
                    for (int c : tied) {
                        double ns = neg_score(neg[c]);
                        negative_scores[c] = ns;
                        if (min_c < 0 || ns < min_neg - NUM_EPS) { min_c = c; min_neg = ns; }
                        else if (std::fabs(ns-min_neg)<=NUM_EPS) {
                            if (c == base_c) min_c = c;
                            else if (min_c != base_c) {
                                double bc = score[c] + lambda_prior * std::log(prior[c]);
                                double bm = score[min_c] + lambda_prior * std::log(prior[min_c]);
                                if (bc > bm + NUM_EPS || (std::fabs(bc-bm)<=NUM_EPS && c < min_c)) min_c = c;
                            }
                        }
                    }

                    prediction = base_c;
                    il.base_negative_score = neg_score(neg[base_c]);
                    il.alternative_class = min_c;
                    if (min_c >= 0) {
                        il.alternative_positive_score = score[min_c];
                        il.alternative_negative_score = min_neg;
                    }
                    if (base_c >= 0 && min_c >= 0 && min_c != base_c &&
                        il.base_negative_score >= M4_NEG_VETO_MIN &&
                        il.base_negative_score - min_neg >= M4_NEG_VETO_DIFF) {
                        il.veto_condition_met = 1;
                        ++out.veto_condition_count;
                        prediction = min_c;
                        il.veto_changed_prediction = 1;
                        ++out.veto_change_count;
                        bool base_correct = (base_c == il.true_class);
                        bool final_correct = (prediction == il.true_class);
                        if (!base_correct && final_correct) {
                            il.veto_change_correct = 1;
                            ++out.veto_correct_change_count;
                        } else if (base_correct && !final_correct) {
                            il.veto_change_incorrect = 1;
                            ++out.veto_incorrect_change_count;
                        }
                    }
                }
            }
        }

        if (collect_rule_evidence) {
            for (int c : labels) {
                ClassEvidenceLog ce;
                ce.class_label = c;
                auto sit = score.find(c);
                if (sit != score.end()) ce.positive_score = sit->second;
                auto nit = negative_scores.find(c);
                if (nit != negative_scores.end()) ce.negative_score = nit->second;
                auto pit = pos.find(c);
                if (pit != pos.end()) {
                    for (const Rule* r : pit->second) {
                        CoveredRuleLog cr;
                        cr.antecedent = r->antecedent;
                        cr.netconf = r->netconf;
                        ce.positive_rules.push_back(std::move(cr));
                    }
                }
                auto nrit = neg.find(c);
                if (nrit != neg.end()) {
                    for (const Rule* r : nrit->second) {
                        CoveredRuleLog cr;
                        cr.antecedent = r->antecedent;
                        cr.netconf = r->netconf;
                        ce.negative_rules.push_back(std::move(cr));
                    }
                }
                il.class_evidence.push_back(std::move(ce));
            }
        }

        il.predicted_class = prediction;
        il.correct = (prediction == il.true_class) ? 1 : 0;
        out.predictions.push_back(prediction);
        if (il.correct) ++out.correct;
        out.instances.push_back(il);
    }

    out.total = static_cast<int>(rows.size());
    out.accuracy = out.total ? out.correct / static_cast<double>(out.total) : 0.0;
    out.macro_f1 = macro_f1(out.truths, out.predictions, labels);
    out.classification_seconds = std::chrono::duration<double>(Clock::now()-t0).count();
    return out;
}


// v5.3.1 optimization: evaluate the complete lambda grid in one pass over the
// validation set. Rule coverage, positive scores, near-tie membership and
// negative-rule coverage are lambda-independent and are therefore computed once
// per instance. Only the prior-adjusted base decision and veto tie-breaking are
// repeated for each lambda.
static std::map<double, EvalResult> classify_lambda_grid(
        const RulesByClass& rules_by_class,
        const Dataset& rows,
        const std::map<int,int>& counts,
        const std::vector<double>& lambda_grid,
        bool collect_rule_evidence = false) {
    if (lambda_grid.empty()) throw std::runtime_error("lambda_grid must not be empty");
    auto t0 = Clock::now();

    std::vector<int> labels;
    int ntrain = 0;
    for (const auto& [c, n] : counts) { labels.push_back(c); ntrain += n; }
    if (labels.empty() || ntrain <= 0) throw std::runtime_error("No class counts available for classification");

    std::map<int,double> prior;
    for (const auto& [c, n] : counts) prior[c] = n / static_cast<double>(ntrain);
    int default_cls = labels.front();
    for (int c : labels) {
        if (prior[c] > prior[default_cls] + NUM_EPS ||
            (std::fabs(prior[c] - prior[default_cls]) <= NUM_EPS && c < default_cls)) default_cls = c;
    }

    std::map<double, EvalResult> results;
    for (double lambda : lambda_grid) results.emplace(lambda, EvalResult{});

    int instance_id = 0;
    for (const auto& row : rows) {
        ++instance_id;
        const int truth = row.back();
        std::set<int> instance(row.begin(), row.end() - 1);

        // Lambda-independent positive coverage.
        std::map<int, std::vector<const Rule*>> pos;
        for (int c : labels) {
            auto it = rules_by_class.find(c);
            if (it == rules_by_class.end()) continue;
            for (const Rule& r : it->second)
                if (!r.negated && exact_cover(r, instance)) pos[c].push_back(&r);
        }
        int exact_total = 0;
        for (const auto& [c, v] : pos) { (void)c; exact_total += static_cast<int>(v.size()); }
        int partial_total = 0;
        int partial_used = 0;
        if (exact_total == 0) {
            for (int c : labels) {
                auto it = rules_by_class.find(c);
                if (it == rules_by_class.end()) continue;
                for (const Rule& r : it->second) {
                    if (!r.negated && one_missing_cover(r, instance)) {
                        pos[c].push_back(&r);
                        ++partial_total;
                    }
                }
            }
            partial_used = partial_total > 0 ? 1 : 0;
        }
        int positive_cover_total = 0;
        std::vector<double> all_pos;
        for (const auto& [c, v] : pos) {
            (void)c;
            positive_cover_total += static_cast<int>(v.size());
            for (const Rule* r : v) all_pos.push_back(r->netconf);
        }

        // Lambda-independent scores and tie set.
        std::map<int,double> score;
        std::vector<int> tied;
        std::map<int,std::vector<const Rule*>> neg;
        std::map<int,double> negative_scores;
        double mu0 = 0.0, best_pos = NEG_INF, second_pos = NEG_INF;
        int best_pos_c = -1;
        long long negative_rules_activated = 0;
        int negative_covered = 0;

        if (!all_pos.empty()) {
            mu0 = std::accumulate(all_pos.begin(), all_pos.end(), 0.0) / all_pos.size();
            for (int c : labels) {
                score[c] = pos_score(pos[c], mu0);
                if (score[c] <= -1e200) continue;
                if (best_pos_c < 0 || score[c] > best_pos + NUM_EPS ||
                    (std::fabs(score[c]-best_pos)<=NUM_EPS && c < best_pos_c)) {
                    best_pos = score[c]; best_pos_c = c;
                }
            }
            for (int c : labels) {
                if (c == best_pos_c || score[c] <= -1e200) continue;
                if (second_pos <= -1e200 || score[c] > second_pos) second_pos = score[c];
            }
            if (best_pos_c >= 0) {
                for (int c : labels)
                    if (score[c] > -1e200 && best_pos - score[c] <= STABLE_TIE_EPS) tied.push_back(c);

                if (tied.size() > 1) {
                    for (int c : tied) {
                        auto it = rules_by_class.find(c);
                        if (it == rules_by_class.end()) continue;
                        for (const Rule& r : it->second) {
                            if (r.negated && exact_cover(r, instance)) {
                                neg[c].push_back(&r);
                                ++negative_rules_activated;
                            }
                        }
                    }
                    negative_covered = negative_rules_activated > 0 ? 1 : 0;
                    for (int c : tied) negative_scores[c] = neg_score(neg[c]);
                }
            }
        }

        // Cheap lambda-dependent decision. No rule is traversed here.
        for (double lambda_prior : lambda_grid) {
            EvalResult& out = results.at(lambda_prior);
            InstanceLog il;
            il.instance_id = instance_id;
            il.true_class = truth;
            if (collect_rule_evidence) il.items.assign(row.begin(), row.end() - 1);
            il.exact_positive_rules = exact_total;
            il.partial_positive_rules = partial_total;
            il.partial_used = partial_used;
            il.positive_cover_total = positive_cover_total;
            il.mu0_positive = mu0;
            il.best_positive_score = best_pos;
            il.second_positive_score = second_pos;
            if (second_pos > -1e200) il.positive_margin = best_pos - second_pos;
            il.tie_size = static_cast<int>(tied.size());

            out.truths.push_back(truth);
            out.exact_positive_rule_count += exact_total;
            out.partial_positive_rule_count += partial_total;
            out.partial_used_count += partial_used;
            out.positive_cover_count += positive_cover_total;
            out.total_tie_classes += il.tie_size;

            int prediction = default_cls;
            if (all_pos.empty() || best_pos_c < 0) {
                il.default_used = 1;
                ++out.default_count;
            } else {
                // v5.4.2: compute the prior-adjusted base winner globally over
                // all classes with positive evidence.  Near-tie membership is
                // lambda-independent diagnostic/PN information and never
                // restricts this positive decision.
                int best_base_c = -1;
                double best_base = NEG_INF;
                for (int c : labels) {
                    auto sit = score.find(c);
                    if (sit == score.end() || sit->second <= -1e200) continue;
                    double base = sit->second + lambda_prior * std::log(prior[c]);
                    if (best_base_c < 0 || base > best_base + NUM_EPS ||
                        (std::fabs(base-best_base)<=NUM_EPS && c < best_base_c)) {
                        best_base = base; best_base_c = c;
                    }
                }

                if (tied.size() <= 1) {
                    prediction = best_base_c;
                    il.base_class = best_base_c;
                    il.base_positive_score = score[best_base_c];
                } else {
                    il.near_tie = 1;
                    il.negative_evaluated = 1;
                    il.negative_covered = negative_covered;
                    il.negative_rules_activated = static_cast<int>(negative_rules_activated);
                    ++out.near_tie_count;
                    ++out.negative_evaluated_count;
                    out.negative_covered_count += negative_covered;
                    out.negative_rule_activation_count += negative_rules_activated;

                    const int base_c = best_base_c;
                    il.base_class = base_c;
                    il.base_positive_score = score[base_c];

                    int min_c = -1;
                    double min_neg = std::numeric_limits<double>::infinity();
                    for (int c : tied) {
                        double ns = negative_scores[c];
                        if (min_c < 0 || ns < min_neg - NUM_EPS) { min_c = c; min_neg = ns; }
                        else if (std::fabs(ns-min_neg)<=NUM_EPS) {
                            if (c == base_c) min_c = c;
                            else if (min_c != base_c) {
                                double bc = score[c] + lambda_prior * std::log(prior[c]);
                                double bm = score[min_c] + lambda_prior * std::log(prior[min_c]);
                                if (bc > bm + NUM_EPS || (std::fabs(bc-bm)<=NUM_EPS && c < min_c)) min_c = c;
                            }
                        }
                    }

                    prediction = base_c;
                    il.base_negative_score = negative_scores[base_c];
                    il.alternative_class = min_c;
                    if (min_c >= 0) {
                        il.alternative_positive_score = score[min_c];
                        il.alternative_negative_score = min_neg;
                    }
                    if (base_c >= 0 && min_c >= 0 && min_c != base_c &&
                        il.base_negative_score >= M4_NEG_VETO_MIN &&
                        il.base_negative_score - min_neg >= M4_NEG_VETO_DIFF) {
                        il.veto_condition_met = 1;
                        ++out.veto_condition_count;
                        prediction = min_c;
                        il.veto_changed_prediction = 1;
                        ++out.veto_change_count;
                        bool base_correct = (base_c == truth);
                        bool final_correct = (prediction == truth);
                        if (!base_correct && final_correct) {
                            il.veto_change_correct = 1;
                            ++out.veto_correct_change_count;
                        } else if (base_correct && !final_correct) {
                            il.veto_change_incorrect = 1;
                            ++out.veto_incorrect_change_count;
                        }
                    }
                }
            }

            if (collect_rule_evidence) {
                for (int c : labels) {
                    ClassEvidenceLog ce;
                    ce.class_label = c;
                    auto sit = score.find(c);
                    if (sit != score.end()) ce.positive_score = sit->second;
                    auto nit = negative_scores.find(c);
                    if (nit != negative_scores.end()) ce.negative_score = nit->second;
                    auto pit = pos.find(c);
                    if (pit != pos.end()) {
                        for (const Rule* r : pit->second) {
                            CoveredRuleLog cr;
                            cr.antecedent = r->antecedent;
                            cr.netconf = r->netconf;
                            ce.positive_rules.push_back(std::move(cr));
                        }
                    }
                    auto nrit = neg.find(c);
                    if (nrit != neg.end()) {
                        for (const Rule* r : nrit->second) {
                            CoveredRuleLog cr;
                            cr.antecedent = r->antecedent;
                            cr.netconf = r->netconf;
                            ce.negative_rules.push_back(std::move(cr));
                        }
                    }
                    il.class_evidence.push_back(std::move(ce));
                }
            }

            il.predicted_class = prediction;
            il.correct = prediction == truth ? 1 : 0;
            out.predictions.push_back(prediction);
            if (il.correct) ++out.correct;
            out.instances.push_back(std::move(il));
        }
    }

    const double total_seconds = std::chrono::duration<double>(Clock::now()-t0).count();
    const double amortized_seconds = total_seconds / static_cast<double>(lambda_grid.size());
    for (auto& [lambda, out] : results) {
        (void)lambda;
        out.total = static_cast<int>(rows.size());
        out.accuracy = out.total ? out.correct / static_cast<double>(out.total) : 0.0;
        out.macro_f1 = macro_f1(out.truths, out.predictions, labels);
        out.classification_seconds = amortized_seconds;
    }
    return results;
}

static double run_miner(const fs::path& miner, const fs::path& train, double sigma, const fs::path& rules,
                        const Variant& v, double min_netconf, const fs::path& log_path) {
    auto t0 = Clock::now();
    std::ostringstream cmd;
    cmd << quote(miner) << ' ' << quote(train) << ' ' << std::setprecision(12) << sigma << ' '
        << quote(rules) << ' ' << v.support_mode << ' ' << v.evidence_mode << ' ' << min_netconf
        << " > " << quote(log_path) << " 2>&1";
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) throw std::runtime_error("Miner failed for " + v.name + ". See " + log_path.string());
    return std::chrono::duration<double>(Clock::now()-t0).count();
}

struct PilotResult {
    bool completed{};
    bool timed_out{};
    int exit_code{};
    double seconds{};
    RuleStatistics rule_stats;
};

static int decoded_exit_code(int rc) {
#if !defined(_WIN32) || defined(__CYGWIN__)
    if (rc == -1) return -1;
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    if (WIFSIGNALED(rc)) return 128 + WTERMSIG(rc);
#endif
    return rc;
}

static PilotResult run_miner_bounded(const fs::path& miner, const fs::path& train, double sigma,
                                     const fs::path& rules, const Variant& v, double min_netconf,
                                     const fs::path& log_path, int time_limit_sec) {
    auto t0 = Clock::now();
    std::ostringstream cmd;
#if defined(__CYGWIN__) || (!defined(_WIN32))
    cmd << "/usr/bin/timeout --signal=TERM --kill-after=5s " << time_limit_sec << "s ";
#endif
    cmd << quote(miner) << ' ' << quote(train) << ' ' << std::setprecision(12) << sigma << ' '
        << quote(rules) << ' ' << v.support_mode << ' ' << v.evidence_mode << ' ' << min_netconf
        << " > " << quote(log_path) << " 2>&1";
    int rc_raw = std::system(cmd.str().c_str());
    PilotResult out;
    out.seconds = std::chrono::duration<double>(Clock::now()-t0).count();
    out.exit_code = decoded_exit_code(rc_raw);
    out.timed_out = (out.exit_code == 124 || out.exit_code == 137 || out.exit_code == 143);
    out.completed = (out.exit_code == 0);
    if (out.completed && fs::exists(rules)) {
        (void)parse_rules(rules, &out.rule_stats);
    }
    return out;
}

static Dataset stratified_fraction(const Dataset& data, double fraction, unsigned seed) {
    std::map<int, Dataset> by_class;
    for (const auto& row : data) {
        if (row.empty()) continue;
        by_class[row.back()].push_back(row);
    }
    std::mt19937 rng(seed);
    Dataset out;
    for (auto& [label, rows] : by_class) {
        std::shuffle(rows.begin(), rows.end(), rng);
        size_t take = static_cast<size_t>(std::ceil(rows.size() * fraction));
        take = std::max<size_t>(1, std::min(take, rows.size()));
        out.insert(out.end(), rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(take));
    }
    std::shuffle(out.begin(), out.end(), rng);
    return out;
}

static bool better_variant(const VariantScore& a, const VariantScore& b) {
    if (a.mean_macro_f1 > b.mean_macro_f1 + NUM_EPS) return true;
    if (b.mean_macro_f1 > a.mean_macro_f1 + NUM_EPS) return false;
    if (a.mean_rules < b.mean_rules - NUM_EPS) return true;
    if (b.mean_rules < a.mean_rules - NUM_EPS) return false;
    // Lambda candidates of the same structural variant reuse exactly the same
    // mined rules. Resolve exact metric ties deterministically in favor of the
    // smaller lambda, i.e. the weaker prior correction. This also lets the
    // selector prefer the no-prior model (lambda=0) when prior correction does
    // not improve the inner-validation criterion.
    if (a.variant.name == b.variant.name) {
        if (a.lambda_prior < b.lambda_prior - NUM_EPS) return true;
        if (b.lambda_prior < a.lambda_prior - NUM_EPS) return false;
        // If predictive performance, model size and prior influence are all
        // identical, prefer the stricter support threshold.
        if (a.sigma > b.sigma + NUM_EPS) return true;
        if (b.sigma > a.sigma + NUM_EPS) return false;
    }
    // Runtime is an audit/computational-cost measurement only; it must not
    // participate in model selection. Remaining structural ties use fixed,
    // deterministic preferences.
    if (a.variant.uses_negatives() != b.variant.uses_negatives()) return !a.variant.uses_negatives();
    if (a.variant.uses_cns() != b.variant.uses_cns()) return !a.variant.uses_cns();
    return a.variant.name < b.variant.name;
}

static void write_instance_logs(std::ofstream& out, const RunMetrics& rm) {
    for (const auto& x : rm.eval.instances) {
        out << rm.phase << ',' << rm.outer_fold << ',' << rm.inner_fold << ',' << rm.variant.name << ','
            << std::setprecision(12) << rm.sigma << ',' << rm.lambda_prior << ',' << x.instance_id << ',' << x.true_class << ','
            << x.predicted_class << ',' << x.correct << ',' << x.default_used << ',' << x.partial_used << ','
            << x.exact_positive_rules << ',' << x.partial_positive_rules << ',' << x.positive_cover_total << ','
            << x.best_positive_score << ',' << x.second_positive_score << ',' << x.positive_margin << ','
            << x.mu0_positive << ',' << x.near_tie << ',' << x.tie_size << ',' << x.negative_evaluated << ','
            << x.negative_covered << ',' << x.negative_rules_activated << ',' << x.base_class << ','
            << x.base_positive_score << ',' << x.base_negative_score << ',' << x.alternative_class << ','
            << x.alternative_positive_score << ',' << x.alternative_negative_score << ','
            << x.veto_condition_met << ',' << x.veto_changed_prediction << ',' << x.veto_change_correct << ','
            << x.veto_change_incorrect << '\n';
    }
    out.flush();
}

static void write_run_summary(std::ofstream& out, const RunMetrics& rm) {
    const EvalResult& e = rm.eval;
    out << rm.phase << ',' << rm.outer_fold << ',' << rm.inner_fold << ',' << rm.variant.name << ','
        << std::setprecision(12) << rm.sigma << ',' << rm.lambda_prior << ',' << rm.training_instances << ',' << rm.validation_instances << ','
        << rm.rule_stats.total_rules << ',' << rm.rule_stats.positive_rules << ',' << rm.rule_stats.negative_rules << ','
        << rm.mining_seconds << ',' << e.classification_seconds << ',' << rm.total_seconds() << ','
        << e.correct << ',' << e.total << ',' << e.accuracy << ',' << e.macro_f1 << ',' << e.default_count << ','
        << e.partial_used_count << ',' << e.exact_positive_rule_count << ',' << e.partial_positive_rule_count << ','
        << e.positive_cover_count << ',' << e.near_tie_count << ',' << e.total_tie_classes << ','
        << e.negative_evaluated_count << ',' << e.negative_covered_count << ',' << e.negative_rule_activation_count << ','
        << e.veto_condition_count << ',' << e.veto_change_count << ',' << e.veto_correct_change_count << ','
        << e.veto_incorrect_change_count << '\n';
    out.flush();
}

static void write_analysis_instances(const fs::path& path,
                                     const RunMetrics& rm,
                                     const fs::path& train_path,
                                     const fs::path& eval_path,
                                     const fs::path& rules_path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot create analysis file: " + path.string());
    out << "ANALISIS DE INSTANCIAS - Auto-DMC-AC / M4d\n"
        << "============================================================\n"
        << "PHASE = " << rm.phase << "\n"
        << "OUTER FOLD = " << rm.outer_fold << "\n"
        << "INNER FOLD = " << rm.inner_fold << "\n"
        << "VARIANT = " << rm.variant.name << "\n"
        << "SUPPORT MODE = " << rm.variant.support_mode << "\n"
        << "EVIDENCE MODE = " << rm.variant.evidence_mode << "\n"
        << "SIGMA = " << std::setprecision(12) << rm.sigma << "\n"
        << "TRAIN = " << train_path.string() << "\n"
        << "EVALUATION = " << eval_path.string() << "\n"
        << "RULES = " << rules_path.string() << "\n"
        << "TOTAL RULES = " << rm.rule_stats.total_rules
        << " | POS = " << rm.rule_stats.positive_rules
        << " | NEG = " << rm.rule_stats.negative_rules << "\n"
        << "MINING TIME = " << rm.mining_seconds << " s\n"
        << "CLASSIFICATION TIME = " << rm.eval.classification_seconds << " s\n"
        << "============================================================\n\n";

    for (const auto& x : rm.eval.instances) {
        out << "INSTANCIA " << x.instance_id << ":\nItems: ";
        for (int item : x.items) out << item << ' ';
        out << "\nClase real: " << x.true_class << "\n\n"
            << "Cobertura positiva:\n"
            << "  cobertura exacta disponible: " << (x.exact_positive_rules > 0 ? "YES" : "NO") << "\n"
            << "  cobertura parcial usada: " << (x.partial_used ? "YES" : "NO") << "\n"
            << "  reglas positivas exactas: " << x.exact_positive_rules << "\n"
            << "  reglas positivas parciales: " << x.partial_positive_rules << "\n"
            << "  cobertura positiva total: " << x.positive_cover_total << "\n\n";

        for (const auto& ce : x.class_evidence) {
            out << "Clase " << ce.class_label << ":\n  POS: ";
            if (ce.positive_rules.empty()) out << "NINGUNA\n";
            else {
                out << "\n";
                for (size_t i = 0; i < ce.positive_rules.size(); ++i) {
                    out << "    #" << (i + 1) << ": {";
                    for (int item : ce.positive_rules[i].antecedent) out << item << ' ';
                    out << "} -> " << ce.class_label
                        << " [Netconf=" << ce.positive_rules[i].netconf << "]\n";
                }
            }
            out << "  Score positivo = ";
            if (ce.positive_score <= -1e200) out << "N/A\n";
            else out << ce.positive_score << "\n";

            out << "  NEG: ";
            if (ce.negative_rules.empty()) out << "NINGUNA\n";
            else {
                out << "\n";
                for (size_t i = 0; i < ce.negative_rules.size(); ++i) {
                    out << "    #" << (i + 1) << ": {";
                    for (int item : ce.negative_rules[i].antecedent) out << item << ' ';
                    out << "} -> neg(" << ce.class_label << ")"
                        << " [Netconf=" << ce.negative_rules[i].netconf << "]\n";
                }
            }
            out << "  Score negativo = " << ce.negative_score << "\n\n";
        }

        out << "Resumen de decisión:\n"
            << "  mu0 positivo = " << x.mu0_positive << "\n"
            << "  mejor score positivo = ";
        if (x.best_positive_score <= -1e200) out << "N/A\n"; else out << x.best_positive_score << "\n";
        out << "  segundo score positivo = ";
        if (x.second_positive_score <= -1e200) out << "N/A\n"; else out << x.second_positive_score << "\n";
        out << "  margen positivo = " << x.positive_margin << "\n"
            << "  near tie = " << (x.near_tie ? "YES" : "NO") << "\n"
            << "  tie size = " << x.tie_size << "\n"
            << "  etapa negativa evaluada = " << (x.negative_evaluated ? "YES" : "NO") << "\n"
            << "  cobertura negativa = " << (x.negative_covered ? "YES" : "NO") << "\n"
            << "  reglas negativas activadas = " << x.negative_rules_activated << "\n"
            << "  clase base = " << x.base_class << "\n"
            << "  score positivo base = ";
        if (x.base_positive_score <= -1e200) out << "N/A\n"; else out << x.base_positive_score << "\n";
        out << "  score negativo base = " << x.base_negative_score << "\n"
            << "  clase alternativa = " << x.alternative_class << "\n"
            << "  score positivo alternativa = ";
        if (x.alternative_positive_score <= -1e200) out << "N/A\n"; else out << x.alternative_positive_score << "\n";
        out << "  score negativo alternativa = " << x.alternative_negative_score << "\n"
            << "  condición de veto = " << (x.veto_condition_met ? "YES" : "NO") << "\n"
            << "  predicción modificada por veto = " << (x.veto_changed_prediction ? "YES" : "NO") << "\n"
            << "  clase asignada = " << x.predicted_class << "\n"
            << "  clase por defecto usada = " << (x.default_used ? "YES" : "NO") << "\n"
            << "  resultado = " << (x.correct ? "CORRECTO" : "INCORRECTO") << "\n"
            << "------------------------------------------------------------\n\n";
    }

    out << "RESUMEN DE LA EJECUCIÓN\n"
        << "Correctos = " << rm.eval.correct << " de " << rm.eval.total << "\n"
        << "Accuracy = " << rm.eval.accuracy << "\n"
        << "Macro-F1 = " << rm.eval.macro_f1 << "\n"
        << "Defaults = " << rm.eval.default_count << "\n"
        << "Instancias con cobertura parcial = " << rm.eval.partial_used_count << "\n"
        << "Reglas positivas exactas activadas = " << rm.eval.exact_positive_rule_count << "\n"
        << "Reglas positivas parciales activadas = " << rm.eval.partial_positive_rule_count << "\n"
        << "Near ties = " << rm.eval.near_tie_count << "\n"
        << "Etapa negativa evaluada = " << rm.eval.negative_evaluated_count << "\n"
        << "Instancias con cobertura negativa = " << rm.eval.negative_covered_count << "\n"
        << "Reglas negativas activadas = " << rm.eval.negative_rule_activation_count << "\n"
        << "Condiciones de veto satisfechas = " << rm.eval.veto_condition_count << "\n"
        << "Cambios por veto = " << rm.eval.veto_change_count << "\n"
        << "Cambios correctos = " << rm.eval.veto_correct_change_count << "\n"
        << "Cambios incorrectos = " << rm.eval.veto_incorrect_change_count << "\n";
}

struct Options {
    fs::path partition_dir;
    fs::path miner;
    fs::path output_dir = "auto_dmc_results";
    // Approximately half-decade log grid. Sigma is selected using inner
    // validation only; dataset-specific manual support values are not needed.
    std::vector<double> sigma_grid{0.30, 0.10, 0.03, 0.01, 0.003, 0.001};
    int sigma_mining_time_limit_sec = 30;
    double min_netconf = 0.0;
    std::vector<double> lambda_grid{0.00, 0.05, 0.10, 0.15, 0.20, 0.30};
    int inner_folds = 3;
    unsigned seed = 20260715;
    int outer_folds = 10;
    std::string analysis_scope = "final";      // none | final | all
    std::string instance_log_scope = "final";  // none | final | all

    // Negative-rule computational guard. The decision uses training data only.
    std::string pn_policy = "auto";             // auto | always | never
    int pn_max_classes = 20;                     // 0 disables this guard
    double pn_memory_budget_mb = 128.0;          // explicit, reproducible budget
    double pn_bytes_per_rule = 256.0;            // conservative stored-rule footprint
    double pn_estimation_safety = 1.25;           // inflation of R+ (C-1)

    // CNS computational guard. All decisions use the outer-training partition only.
    std::string cns_policy = "auto";             // auto | always | never
    double cns_max_support_relaxation = 12.0;     // trigger bounded pilot above this ratio
    int cns_min_absolute_support = 5;              // hard structural floor
    double cns_pilot_fraction = 0.10;              // stratified fraction of outer training
    int cns_pilot_time_limit_sec = 30;             // hard timeout
    double cns_max_estimated_time_sec = 300.0;      // per mining run
    double cns_memory_budget_mb = 1024.0;
    double cns_bytes_per_rule = 256.0;
    double cns_estimation_safety = 2.0;
};

static void usage(const char* exe) {
    std::cerr << "Usage: " << exe << " --partition-dir DIR --miner PATH [--output-dir DIR]"
              << " [--sigma VALUE | --sigma-grid 0.001,0.003,0.01,0.03,0.10,0.30]"
              << " [--sigma-mining-time-limit-sec 30] [--min-netconf 0.0]"
              << " [--lambda-grid 0.00,0.05,0.10,0.15,0.20,0.30] [--inner-folds 3] [--seed N] [--outer-folds 10]"
              << " [--analysis-scope none|final|all] [--instance-log-scope none|final|all]"
              << " [--pn-policy auto|always|never] [--pn-max-classes 20]"
              << " [--pn-memory-budget-mb 128] [--pn-bytes-per-rule 256]"
              << " [--pn-estimation-safety 1.25]"
              << " [--cns-policy auto|always|never] [--cns-max-support-relaxation 12]"
              << " [--cns-min-absolute-support 5] [--cns-pilot-fraction 0.10]"
              << " [--cns-pilot-time-limit-sec 30] [--cns-max-estimated-time-sec 300]"
              << " [--cns-memory-budget-mb 1024] [--cns-bytes-per-rule 256]"
              << " [--cns-estimation-safety 2.0]\n";
}

static std::vector<double> parse_lambda_grid(const std::string& text) {
    std::vector<double> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        double v = std::stod(token);
        if (v < 0.0 || !std::isfinite(v)) throw std::runtime_error("lambda values must be finite and >= 0");
        values.push_back(v);
    }
    if (values.empty()) throw std::runtime_error("--lambda-grid must contain at least one value");
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end(), [](double a, double b){ return std::fabs(a-b) <= NUM_EPS; }), values.end());
    return values;
}

static std::vector<double> parse_sigma_grid(const std::string& text) {
    std::vector<double> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        double v = std::stod(token);
        if (!(v > 0.0 && v <= 1.0) || !std::isfinite(v))
            throw std::runtime_error("sigma values must be finite and in (0,1]");
        values.push_back(v);
    }
    if (values.empty()) throw std::runtime_error("--sigma-grid must contain at least one value");
    // Evaluate from stricter to more permissive support. This order permits a
    // resource cutoff once global positive mining times out: every lower
    // support has a superset frequent-pattern search space.
    std::sort(values.begin(), values.end(), std::greater<double>());
    values.erase(std::unique(values.begin(), values.end(), [](double a, double b){ return std::fabs(a-b) <= NUM_EPS; }), values.end());
    return values;
}

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("Missing value for " + a); return argv[i]; };
        if (a == "--partition-dir") o.partition_dir = need();
        else if (a == "--miner") o.miner = need();
        else if (a == "--output-dir") o.output_dir = need();
        else if (a == "--sigma") o.sigma_grid = parse_sigma_grid(need());
        else if (a == "--sigma-grid") o.sigma_grid = parse_sigma_grid(need());
        else if (a == "--sigma-mining-time-limit-sec") o.sigma_mining_time_limit_sec = std::stoi(need());
        else if (a == "--min-netconf") o.min_netconf = std::stod(need());
        else if (a == "--lambda-grid") o.lambda_grid = parse_lambda_grid(need());
        else if (a == "--inner-folds") o.inner_folds = std::stoi(need());
        else if (a == "--seed") o.seed = static_cast<unsigned>(std::stoul(need()));
        else if (a == "--outer-folds") o.outer_folds = std::stoi(need());
        else if (a == "--analysis-scope") o.analysis_scope = need();
        else if (a == "--instance-log-scope") o.instance_log_scope = need();
        else if (a == "--pn-policy") o.pn_policy = need();
        else if (a == "--pn-max-classes") o.pn_max_classes = std::stoi(need());
        else if (a == "--pn-memory-budget-mb") o.pn_memory_budget_mb = std::stod(need());
        else if (a == "--pn-bytes-per-rule") o.pn_bytes_per_rule = std::stod(need());
        else if (a == "--pn-estimation-safety") o.pn_estimation_safety = std::stod(need());
        else if (a == "--cns-policy") o.cns_policy = need();
        else if (a == "--cns-max-support-relaxation") o.cns_max_support_relaxation = std::stod(need());
        else if (a == "--cns-min-absolute-support") o.cns_min_absolute_support = std::stoi(need());
        else if (a == "--cns-pilot-fraction") o.cns_pilot_fraction = std::stod(need());
        else if (a == "--cns-pilot-time-limit-sec") o.cns_pilot_time_limit_sec = std::stoi(need());
        else if (a == "--cns-max-estimated-time-sec") o.cns_max_estimated_time_sec = std::stod(need());
        else if (a == "--cns-memory-budget-mb") o.cns_memory_budget_mb = std::stod(need());
        else if (a == "--cns-bytes-per-rule") o.cns_bytes_per_rule = std::stod(need());
        else if (a == "--cns-estimation-safety") o.cns_estimation_safety = std::stod(need());
        else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
        else throw std::runtime_error("Unknown argument: " + a);
    }
    if (o.partition_dir.empty() || o.miner.empty()) throw std::runtime_error("--partition-dir and --miner are required");
    if (o.sigma_grid.empty()) throw std::runtime_error("sigma grid must not be empty");
    if (o.sigma_mining_time_limit_sec < 1) throw std::runtime_error("--sigma-mining-time-limit-sec must be >= 1");
    if (o.inner_folds < 2) throw std::runtime_error("--inner-folds must be >= 2");
    auto valid_scope = [](const std::string& x) { return x == "none" || x == "final" || x == "all"; };
    if (!valid_scope(o.analysis_scope)) throw std::runtime_error("--analysis-scope must be none, final, or all");
    if (!valid_scope(o.instance_log_scope)) throw std::runtime_error("--instance-log-scope must be none, final, or all");
    if (o.min_netconf < 0.0) throw std::runtime_error("--min-netconf must be >= 0.0");
    if (!(o.pn_policy == "auto" || o.pn_policy == "always" || o.pn_policy == "never"))
        throw std::runtime_error("--pn-policy must be auto, always, or never");
    if (o.pn_max_classes < 0) throw std::runtime_error("--pn-max-classes must be >= 0");
    if (o.pn_memory_budget_mb <= 0.0) throw std::runtime_error("--pn-memory-budget-mb must be > 0");
    if (o.pn_bytes_per_rule <= 0.0) throw std::runtime_error("--pn-bytes-per-rule must be > 0");
    if (o.pn_estimation_safety < 1.0) throw std::runtime_error("--pn-estimation-safety must be >= 1.0");
    if (!(o.cns_policy == "auto" || o.cns_policy == "always" || o.cns_policy == "never"))
        throw std::runtime_error("--cns-policy must be auto, always, or never");
    if (o.cns_max_support_relaxation <= 0.0) throw std::runtime_error("--cns-max-support-relaxation must be > 0");
    if (o.cns_min_absolute_support < 1) throw std::runtime_error("--cns-min-absolute-support must be >= 1");
    if (!(o.cns_pilot_fraction > 0.0 && o.cns_pilot_fraction <= 1.0)) throw std::runtime_error("--cns-pilot-fraction must be in (0,1]");
    if (o.cns_pilot_time_limit_sec < 1) throw std::runtime_error("--cns-pilot-time-limit-sec must be >= 1");
    if (o.cns_max_estimated_time_sec <= 0.0) throw std::runtime_error("--cns-max-estimated-time-sec must be > 0");
    if (o.cns_memory_budget_mb <= 0.0) throw std::runtime_error("--cns-memory-budget-mb must be > 0");
    if (o.cns_bytes_per_rule <= 0.0) throw std::runtime_error("--cns-bytes-per-rule must be > 0");
    if (o.cns_estimation_safety < 1.0) throw std::runtime_error("--cns-estimation-safety must be >= 1.0");
    return o;
}

static LogWriters open_logs(const fs::path& output_dir) {
    LogWriters w;
    w.instance.open(output_dir / "instance_log.csv");
    w.run_summary.open(output_dir / "fold_variant_summary.csv");
    w.selection.open(output_dir / "selection_log.csv");
    w.outer_summary.open(output_dir / "dataset_summary.csv");
    w.pn_guard.open(output_dir / "pn_guard.csv");
    w.cns_guard.open(output_dir / "cns_guard.csv");
    if (!w.instance || !w.run_summary || !w.selection || !w.outer_summary || !w.pn_guard || !w.cns_guard)
        throw std::runtime_error("Cannot create one or more log files in " + output_dir.string());

    w.instance << "phase,outer_fold,inner_fold,variant,sigma,lambda_prior,instance_id,true_class,predicted_class,correct,default_used,partial_used,exact_positive_rules,partial_positive_rules,positive_cover_total,best_positive_score,second_positive_score,positive_margin,mu0_positive,near_tie,tie_size,negative_evaluated,negative_covered,negative_rules_activated,base_class,base_positive_score,base_negative_score,alternative_class,alternative_positive_score,alternative_negative_score,veto_condition_met,veto_changed_prediction,veto_change_correct,veto_change_incorrect\n";
    w.run_summary << "phase,outer_fold,inner_fold,variant,sigma,lambda_prior,training_instances,validation_instances,total_rules,positive_rules,negative_rules,mining_seconds,classification_seconds,total_seconds,correct,total,accuracy,macro_f1,default_count,partial_used_count,exact_positive_rule_count,partial_positive_rule_count,positive_cover_count,near_tie_count,total_tie_classes,negative_evaluated_count,negative_covered_count,negative_rule_activation_count,veto_condition_count,veto_change_count,veto_correct_change_count,veto_incorrect_change_count\n";
    w.selection << "outer_fold,rank,variant,sigma,lambda_prior,mean_macro_f1,sd_macro_f1,mean_accuracy,mean_rules,mean_positive_rules,mean_negative_rules,mean_mining_seconds,mean_classification_seconds,mean_total_seconds,selected\n";
    w.pn_guard << "outer_fold,sigma,support_mode,policy,class_count,max_positive_rules,estimated_negative_candidates,rule_budget,class_guard_pass,budget_guard_pass,decision,reason\n";
    w.cns_guard << "outer_fold,sigma,policy,training_instances,class_count,global_abs_support,min_class_abs_support,support_relaxation,pilot_fraction,pilot_instances,pilot_seconds,pilot_exit_code,pilot_rules,estimated_seconds,estimated_rules,estimated_memory_mb,decision,reason\n";
    w.outer_summary << "outer_fold,selected_variant,sigma,selected_lambda_prior,inner_macro_f1,inner_macro_f1_sd,inner_accuracy,mean_rules,mean_positive_rules,mean_negative_rules,mean_mining_seconds,mean_classification_seconds,selection_seconds,final_mining_seconds,final_classification_seconds,outer_accuracy,outer_macro_f1,correct,total,default_count,partial_used_count,exact_positive_rule_count,partial_positive_rule_count,positive_cover_count,near_tie_count,total_tie_classes,negative_evaluated_count,negative_covered_count,negative_rule_activation_count,veto_condition_count,veto_change_count,veto_correct_change_count,veto_incorrect_change_count\n";
    return w;
}

int main(int argc, char** argv) {
    try {
        auto dataset_t0 = Clock::now();
        Options opt = parse_args(argc, argv);
        fs::create_directories(opt.output_dir);
        LogWriters logs = open_logs(opt.output_dir);

        {
            std::ofstream cfg(opt.output_dir / "run_configuration.txt");
            cfg << std::setprecision(12)
                << "partition_dir=" << fs::absolute(opt.partition_dir).string() << '\n'
                << "miner=" << fs::absolute(opt.miner).string() << '\n'
                << "output_dir=" << fs::absolute(opt.output_dir).string() << '\n'
                << "sigma_grid=";
            for (size_t i = 0; i < opt.sigma_grid.size(); ++i) { if (i) cfg << ','; cfg << opt.sigma_grid[i]; }
            cfg << '\n'
                << "sigma_mining_time_limit_sec=" << opt.sigma_mining_time_limit_sec << '\n'
                << "min_netconf=" << opt.min_netconf << '\n';
            cfg << "lambda_grid=";
            for (size_t i = 0; i < opt.lambda_grid.size(); ++i) { if (i) cfg << ','; cfg << opt.lambda_grid[i]; }
            cfg << '\n'
                << "inner_folds=" << opt.inner_folds << '\n'
                << "outer_folds=" << opt.outer_folds << '\n'
                << "seed=" << opt.seed << '\n'
                << "analysis_scope=" << opt.analysis_scope << '\n'
                << "instance_log_scope=" << opt.instance_log_scope << '\n'
                << "pn_policy=" << opt.pn_policy << '\n'
                << "pn_max_classes=" << opt.pn_max_classes << '\n'
                << "pn_memory_budget_mb=" << opt.pn_memory_budget_mb << '\n'
                << "pn_bytes_per_rule=" << opt.pn_bytes_per_rule << '\n'
                << "pn_estimation_safety=" << opt.pn_estimation_safety << '\n'
                << "cns_policy=" << opt.cns_policy << '\n'
                << "cns_max_support_relaxation=" << opt.cns_max_support_relaxation << '\n'
                << "cns_min_absolute_support=" << opt.cns_min_absolute_support << '\n'
                << "cns_pilot_fraction=" << opt.cns_pilot_fraction << '\n'
                << "cns_pilot_time_limit_sec=" << opt.cns_pilot_time_limit_sec << '\n'
                << "cns_max_estimated_time_sec=" << opt.cns_max_estimated_time_sec << '\n'
                << "cns_memory_budget_mb=" << opt.cns_memory_budget_mb << '\n'
                << "cns_bytes_per_rule=" << opt.cns_bytes_per_rule << '\n'
                << "cns_estimation_safety=" << opt.cns_estimation_safety << '\n';
        }
        std::cout << std::setprecision(12)
                  << "Configuration: sigma_grid=";
        for (size_t i = 0; i < opt.sigma_grid.size(); ++i) std::cout << (i ? "," : "") << opt.sigma_grid[i];
        std::cout << " sigma_mining_time_limit_sec=" << opt.sigma_mining_time_limit_sec
                  << " min_netconf=" << opt.min_netconf
                  << " pn_policy=" << opt.pn_policy
                  << " pn_max_classes=" << opt.pn_max_classes
                  << " pn_memory_budget_mb=" << opt.pn_memory_budget_mb
                  << " cns_policy=" << opt.cns_policy << " lambda_grid=";
        for (size_t i = 0; i < opt.lambda_grid.size(); ++i) std::cout << (i ? "," : "") << opt.lambda_grid[i];
        std::cout << '\n';

        const std::vector<Variant> variants = {
            {"global_p", "global", "p"}, {"global_pn", "global", "pn"},
            {"cns_p", "cns", "p"}, {"cns_pn", "cns", "pn"}
        };

        double acc_sum = 0.0, f1_sum = 0.0;
        double selection_time_sum = 0.0, final_mining_sum = 0.0, final_classification_sum = 0.0;
        std::map<std::string,int> selected_counts;

        for (int outer = 1; outer <= opt.outer_folds; ++outer) {
            auto selection_t0 = Clock::now();
            fs::path outer_train_path = opt.partition_dir / ("Dataset" + std::to_string(outer) + ".dat");
            fs::path outer_test_path = opt.partition_dir / (std::to_string(outer) + ".dat");
            Dataset outer_train = read_dataset(outer_train_path);
            Dataset outer_test = read_dataset(outer_test_path);
            auto folds = stratified_folds(outer_train, opt.inner_folds, opt.seed + static_cast<unsigned>(outer));
            fs::path fold_dir = opt.output_dir / ("outer_" + std::to_string(outer));
            fs::create_directories(fold_dir);
            std::vector<VariantScore> scores;
            const auto outer_class_counts = class_counts(outer_train);
            const int class_count = static_cast<int>(outer_class_counts.size());
            const double rule_budget = std::floor(opt.pn_memory_budget_mb * 1024.0 * 1024.0 / opt.pn_bytes_per_rule);

            bool lower_supports_infeasible = false;
            for (double sigma : opt.sigma_grid) {
            std::map<std::string, double> max_positive_rules_by_support;
            const int global_abs_support = std::max(1, static_cast<int>(std::ceil(sigma * outer_train.size())));
            int min_class_abs_support = std::numeric_limits<int>::max();
            for (const auto& [label, count] : outer_class_counts)
                min_class_abs_support = std::min(min_class_abs_support, std::max(1, static_cast<int>(std::ceil(sigma * count))));
            if (min_class_abs_support == std::numeric_limits<int>::max()) min_class_abs_support = 1;
            const double support_relaxation = global_abs_support / static_cast<double>(min_class_abs_support);
            bool cns_allowed = true;
            std::string cns_reason = "forced_by_policy";
            PilotResult cns_pilot;
            double cns_estimated_seconds = 0.0, cns_estimated_rules = 0.0, cns_estimated_memory_mb = 0.0;
            size_t cns_pilot_instances = 0;

            if (opt.cns_policy == "never") {
                cns_allowed = false; cns_reason = "disabled_by_policy";
            } else if (opt.cns_policy == "always") {
                cns_allowed = true; cns_reason = "forced_by_policy";
            } else if (min_class_abs_support < opt.cns_min_absolute_support) {
                cns_allowed = false; cns_reason = "class_support_below_structural_minimum";
            } else if (support_relaxation <= opt.cns_max_support_relaxation) {
                cns_allowed = true; cns_reason = "within_structural_guard";
            } else {
                std::ostringstream sigma_name_ss;
                sigma_name_ss << std::setprecision(8) << sigma;
                fs::path pilot_dir = fold_dir / ("sigma_" + sigma_name_ss.str()) / "cns_guard_pilot";
                fs::create_directories(pilot_dir);
                Dataset pilot_data = stratified_fraction(outer_train, opt.cns_pilot_fraction,
                                                         opt.seed + 100000u + static_cast<unsigned>(outer));
                cns_pilot_instances = pilot_data.size();
                fs::path pilot_train = pilot_dir / "train.dat";
                fs::path pilot_rules = pilot_dir / "rules.dat";
                write_dataset(pilot_train, pilot_data);
                Variant pilot_variant{"cns_p_pilot", "cns", "p"};
                cns_pilot = run_miner_bounded(opt.miner, pilot_train, sigma, pilot_rules, pilot_variant,
                                              opt.min_netconf, pilot_dir / "miner.log",
                                              opt.cns_pilot_time_limit_sec);
                if (!cns_pilot.completed) {
                    cns_allowed = false;
                    cns_reason = cns_pilot.timed_out ? "pilot_timeout" : "pilot_backend_error";
                } else {
                    const double scale = outer_train.empty() ? 1.0 : outer_train.size() / static_cast<double>(pilot_data.size());
                    cns_estimated_seconds = cns_pilot.seconds * scale * opt.cns_estimation_safety;
                    cns_estimated_rules = cns_pilot.rule_stats.total_rules * scale * opt.cns_estimation_safety;
                    cns_estimated_memory_mb = cns_estimated_rules * opt.cns_bytes_per_rule / (1024.0 * 1024.0);
                    if (cns_estimated_seconds > opt.cns_max_estimated_time_sec) {
                        cns_allowed = false; cns_reason = "estimated_runtime_budget_exceeded";
                    } else if (cns_estimated_memory_mb > opt.cns_memory_budget_mb) {
                        cns_allowed = false; cns_reason = "estimated_memory_budget_exceeded";
                    } else {
                        cns_allowed = true; cns_reason = "pilot_within_budget";
                    }
                }
            }

            logs.cns_guard << outer << ',' << sigma << ',' << opt.cns_policy << ',' << outer_train.size() << ',' << class_count << ','
                           << global_abs_support << ',' << min_class_abs_support << ',' << support_relaxation << ','
                           << opt.cns_pilot_fraction << ',' << cns_pilot_instances << ',' << cns_pilot.seconds << ','
                           << cns_pilot.exit_code << ',' << cns_pilot.rule_stats.total_rules << ','
                           << cns_estimated_seconds << ',' << cns_estimated_rules << ',' << cns_estimated_memory_mb << ','
                           << (cns_allowed ? "EVALUATE" : "SKIP") << ',' << cns_reason << '\n';
            logs.cns_guard.flush();
            std::cout << "CNS guard outer=" << outer << " sigma=" << sigma << " global_abs=" << global_abs_support
                      << " min_class_abs=" << min_class_abs_support << " relaxation=" << support_relaxation
                      << " decision=" << (cns_allowed ? "EVALUATE" : "SKIP")
                      << " reason=" << cns_reason << '\n';

            for (const Variant& v : variants) {
                if (v.uses_cns() && !cns_allowed) continue;
                if (v.uses_negatives()) {
                    const double max_pos = max_positive_rules_by_support[v.support_mode];
                    const double estimated_neg = std::ceil(max_pos * std::max(0, class_count - 1) * opt.pn_estimation_safety);
                    const bool class_ok = (opt.pn_max_classes == 0 || class_count <= opt.pn_max_classes);
                    const bool budget_ok = estimated_neg <= rule_budget;
                    bool allow = false;
                    std::string reason;
                    if (opt.pn_policy == "always") { allow = true; reason = "forced_by_policy"; }
                    else if (opt.pn_policy == "never") { allow = false; reason = "disabled_by_policy"; }
                    else if (!class_ok) { allow = false; reason = "class_count_guard"; }
                    else if (!budget_ok) { allow = false; reason = "memory_derived_rule_budget"; }
                    else { allow = true; reason = "within_auto_budget"; }

                    logs.pn_guard << outer << ',' << sigma << ',' << v.support_mode << ',' << opt.pn_policy << ',' << class_count << ','
                                  << max_pos << ',' << estimated_neg << ',' << rule_budget << ','
                                  << (class_ok ? 1 : 0) << ',' << (budget_ok ? 1 : 0) << ','
                                  << (allow ? "EVALUATE" : "SKIP") << ',' << reason << '\n';
                    logs.pn_guard.flush();
                    std::cout << "PN guard outer=" << outer << " sigma=" << sigma << " support=" << v.support_mode
                              << " classes=" << class_count << " max_R+=" << max_pos
                              << " estimated_R-=" << estimated_neg << " budget=" << rule_budget
                              << " decision=" << (allow ? "EVALUATE" : "SKIP")
                              << " reason=" << reason << '\n';
                    if (!allow) continue;
                }
                struct LambdaAccumulator {
                    std::vector<double> f1s, accs, class_times, total_times;
                };
                std::map<double, LambdaAccumulator> lambda_acc;
                std::vector<double> rules_n, pos_rules_n, neg_rules_n, mine_times;
                bool candidate_timed_out = false;
                for (int inner = 0; inner < opt.inner_folds; ++inner) {
                    Dataset train, valid = folds[static_cast<size_t>(inner)];
                    for (int j = 0; j < opt.inner_folds; ++j) if (j != inner)
                        train.insert(train.end(), folds[static_cast<size_t>(j)].begin(), folds[static_cast<size_t>(j)].end());

                    std::ostringstream sigma_name_ss;
                    sigma_name_ss << std::setprecision(8) << sigma;
                    fs::path inner_dir = fold_dir / ("sigma_" + sigma_name_ss.str()) / (v.name + "_inner_" + std::to_string(inner+1));
                    fs::create_directories(inner_dir);
                    fs::path train_path = inner_dir / "train.dat";
                    fs::path valid_path = inner_dir / "valid.dat";
                    fs::path rules_path = inner_dir / "rules.dat";
                    write_dataset(train_path, train);
                    write_dataset(valid_path, valid);

                    PilotResult bounded = run_miner_bounded(opt.miner, train_path, sigma, rules_path, v,
                                                            opt.min_netconf, inner_dir / "miner.log",
                                                            opt.sigma_mining_time_limit_sec);
                    if (!bounded.completed) {
                        std::cout << "Sigma candidate skipped: outer=" << outer << " sigma=" << sigma
                                  << " variant=" << v.name << " inner=" << (inner + 1)
                                  << " reason=" << (bounded.timed_out ? "mining_timeout" : "miner_error") << '\n';
                        rules_n.clear();
                        candidate_timed_out = bounded.timed_out;
                        break;
                    }
                    double mining_seconds = bounded.seconds;
                    RuleStatistics rule_stats;
                    RulesByClass rules = parse_rules(rules_path, &rule_stats);
                    rules_n.push_back(rule_stats.total_rules);
                    pos_rules_n.push_back(rule_stats.positive_rules);
                    neg_rules_n.push_back(rule_stats.negative_rules);
                    mine_times.push_back(mining_seconds);

                    const auto inner_counts = class_counts(train);
                    auto grid_eval = classify_lambda_grid(rules, valid, inner_counts, opt.lambda_grid,
                                                          opt.analysis_scope == "all");
                    for (double lambda_prior : opt.lambda_grid) {
                        RunMetrics rm;
                        rm.variant = v; rm.lambda_prior = lambda_prior; rm.outer_fold = outer; rm.inner_fold = inner+1; rm.phase = "inner";
                        rm.sigma = sigma; rm.training_instances = static_cast<int>(train.size());
                        rm.validation_instances = static_cast<int>(valid.size());
                        rm.mining_seconds = mining_seconds; rm.rule_stats = rule_stats;
                        rm.eval = std::move(grid_eval.at(lambda_prior));
                        write_run_summary(logs.run_summary, rm);
                        if (opt.instance_log_scope == "all") write_instance_logs(logs.instance, rm);
                        if (opt.analysis_scope == "all") {
                            fs::path lambda_dir = inner_dir / ("lambda_" + std::to_string(lambda_prior));
                            fs::create_directories(lambda_dir);
                            write_analysis_instances(lambda_dir / "AnalisisInstancias.dat", rm, train_path, valid_path, rules_path);
                        }

                        std::ofstream pred(inner_dir / ("predictions_lambda_" + std::to_string(lambda_prior) + ".csv"));
                        pred << "instance,true_class,predicted_class,correct\n";
                        for (const auto& x : rm.eval.instances)
                            pred << x.instance_id << ',' << x.true_class << ',' << x.predicted_class << ',' << x.correct << '\n';

                        auto& a = lambda_acc[lambda_prior];
                        a.f1s.push_back(rm.eval.macro_f1); a.accs.push_back(rm.eval.accuracy);
                        a.class_times.push_back(rm.eval.classification_seconds);
                        a.total_times.push_back(mining_seconds + rm.eval.classification_seconds);
                    }
                }
                if (rules_n.size() != static_cast<size_t>(opt.inner_folds)) {
                    if (v.name == "global_p" && candidate_timed_out) {
                        lower_supports_infeasible = true;
                        std::cout << "Lower sigma candidates rejected in outer=" << outer
                                  << " after global_p timeout at sigma=" << sigma << '\n';
                        break;
                    }
                    continue;
                }
                for (double lambda_prior : opt.lambda_grid) {
                    const auto& a = lambda_acc.at(lambda_prior);
                    scores.push_back({v, sigma, lambda_prior, mean(a.f1s), sample_sd(a.f1s), mean(a.accs), mean(rules_n), mean(pos_rules_n),
                                      mean(neg_rules_n), mean(mine_times), mean(a.class_times), mean(a.total_times)});
                }
                if (!v.uses_negatives() && !pos_rules_n.empty()) {
                    max_positive_rules_by_support[v.support_mode] =
                        *std::max_element(pos_rules_n.begin(), pos_rules_n.end());
                }
            }
            if (lower_supports_infeasible) break;
            }
            if (scores.empty()) throw std::runtime_error("No variant was evaluated in outer fold " + std::to_string(outer));

            std::sort(scores.begin(), scores.end(), better_variant);
            VariantScore best = scores.front();
            selected_counts[best.variant.name + "@sigma=" + std::to_string(best.sigma) + "@lambda=" + std::to_string(best.lambda_prior)]++;
            for (size_t rank = 0; rank < scores.size(); ++rank) {
                const auto& s = scores[rank];
                logs.selection << outer << ',' << (rank+1) << ',' << s.variant.name << ',' << std::setprecision(12)
                               << s.sigma << ',' << s.lambda_prior << ',' << s.mean_macro_f1 << ',' << s.sd_macro_f1 << ',' << s.mean_accuracy << ','
                               << s.mean_rules << ',' << s.mean_positive_rules << ',' << s.mean_negative_rules << ','
                               << s.mean_mining_seconds << ',' << s.mean_classification_seconds << ',' << s.mean_total_seconds << ','
                               << (rank == 0 ? 1 : 0) << '\n';
            }
            logs.selection.flush();
            double selection_seconds = std::chrono::duration<double>(Clock::now()-selection_t0).count();
            selection_time_sum += selection_seconds;

            fs::path final_dir = fold_dir / "final";
            fs::create_directories(final_dir);
            fs::path final_train = final_dir / "train.dat";
            fs::path final_test = final_dir / "test.dat";
            fs::path final_rules = final_dir / "rules.dat";
            write_dataset(final_train, outer_train);
            write_dataset(final_test, outer_test);

            RunMetrics final_rm;
            final_rm.variant = best.variant; final_rm.lambda_prior = best.lambda_prior; final_rm.outer_fold = outer; final_rm.inner_fold = 0; final_rm.phase = "outer";
            final_rm.sigma = best.sigma; final_rm.training_instances = static_cast<int>(outer_train.size());
            final_rm.validation_instances = static_cast<int>(outer_test.size());
            final_rm.mining_seconds = run_miner(opt.miner, final_train, best.sigma, final_rules, best.variant,
                                                opt.min_netconf, final_dir / "miner.log");
            RulesByClass final_rule_set = parse_rules(final_rules, &final_rm.rule_stats);
            final_rm.eval = classify(final_rule_set, outer_test, class_counts(outer_train), best.lambda_prior,
                                     opt.analysis_scope == "all" || opt.analysis_scope == "final");
            write_run_summary(logs.run_summary, final_rm);
            if (opt.instance_log_scope == "all" || opt.instance_log_scope == "final")
                write_instance_logs(logs.instance, final_rm);
            if (opt.analysis_scope == "all" || opt.analysis_scope == "final")
                write_analysis_instances(final_dir / "AnalisisInstancias.dat", final_rm, final_train, final_test, final_rules);

            std::ofstream pred(final_dir / "predictions.csv");
            pred << "instance,true_class,predicted_class,correct,default_used,partial_used,positive_cover_total,near_tie,tie_size,negative_evaluated,negative_covered,negative_rules_activated,veto_changed_prediction\n";
            for (const auto& x : final_rm.eval.instances) {
                pred << x.instance_id << ',' << x.true_class << ',' << x.predicted_class << ',' << x.correct << ','
                     << x.default_used << ',' << x.partial_used << ',' << x.positive_cover_total << ',' << x.near_tie << ','
                     << x.tie_size << ',' << x.negative_evaluated << ',' << x.negative_covered << ','
                     << x.negative_rules_activated << ',' << x.veto_changed_prediction << '\n';
            }

            const EvalResult& e = final_rm.eval;
            logs.outer_summary << outer << ',' << best.variant.name << ',' << std::setprecision(12) << best.sigma << ',' << best.lambda_prior << ','
                << best.mean_macro_f1 << ',' << best.sd_macro_f1 << ',' << best.mean_accuracy << ',' << best.mean_rules << ','
                << best.mean_positive_rules << ',' << best.mean_negative_rules << ',' << best.mean_mining_seconds << ','
                << best.mean_classification_seconds << ',' << selection_seconds << ',' << final_rm.mining_seconds << ','
                << e.classification_seconds << ',' << e.accuracy << ',' << e.macro_f1 << ',' << e.correct << ',' << e.total << ','
                << e.default_count << ',' << e.partial_used_count << ',' << e.exact_positive_rule_count << ','
                << e.partial_positive_rule_count << ',' << e.positive_cover_count << ',' << e.near_tie_count << ','
                << e.total_tie_classes << ',' << e.negative_evaluated_count << ',' << e.negative_covered_count << ','
                << e.negative_rule_activation_count << ',' << e.veto_condition_count << ',' << e.veto_change_count << ','
                << e.veto_correct_change_count << ',' << e.veto_incorrect_change_count << '\n';
            logs.outer_summary.flush();

            acc_sum += e.accuracy; f1_sum += e.macro_f1;
            final_mining_sum += final_rm.mining_seconds;
            final_classification_sum += e.classification_seconds;
            std::cout << "Outer fold " << outer << ": selected=" << best.variant.name
                      << " sigma=" << best.sigma << " lambda=" << best.lambda_prior
                      << " inner_macroF1=" << std::fixed << std::setprecision(4) << best.mean_macro_f1
                      << " outer_accuracy=" << e.accuracy << " outer_macroF1=" << e.macro_f1 << '\n';
        }

        double dataset_seconds = std::chrono::duration<double>(Clock::now()-dataset_t0).count();
        std::ofstream report(opt.output_dir / "auto_dmc_report.txt");
        report << std::fixed << std::setprecision(6)
               << "Sigma grid: ";
        for (size_t i = 0; i < opt.sigma_grid.size(); ++i) report << (i ? "," : "") << opt.sigma_grid[i];
        report << '\n'
               << "Sigma mining time limit per inner run (s): " << opt.sigma_mining_time_limit_sec << '\n'
               << "Inner folds: " << opt.inner_folds << '\n'
               << "Lambda grid: ";
        for (size_t i = 0; i < opt.lambda_grid.size(); ++i) report << (i ? "," : "") << opt.lambda_grid[i];
        report << '\n'
               << "PN policy: " << opt.pn_policy << '\n'
               << "PN max classes: " << opt.pn_max_classes << '\n'
               << "PN memory budget (MiB): " << opt.pn_memory_budget_mb << '\n'
               << "PN bytes per rule: " << opt.pn_bytes_per_rule << '\n'
               << "PN estimation safety: " << opt.pn_estimation_safety << '\n'
               << "CNS policy: " << opt.cns_policy << '\n'
               << "CNS max support relaxation: " << opt.cns_max_support_relaxation << '\n'
               << "CNS minimum absolute support: " << opt.cns_min_absolute_support << '\n'
               << "CNS pilot fraction: " << opt.cns_pilot_fraction << '\n'
               << "CNS pilot timeout (s): " << opt.cns_pilot_time_limit_sec << '\n'
               << "CNS max estimated time (s): " << opt.cns_max_estimated_time_sec << '\n'
               << "CNS memory budget (MiB): " << opt.cns_memory_budget_mb << '\n'
               << "CNS bytes per rule: " << opt.cns_bytes_per_rule << '\n'
               << "CNS estimation safety: " << opt.cns_estimation_safety << '\n'
               << "Mean outer accuracy: " << acc_sum / opt.outer_folds << '\n'
               << "Mean outer macro-F1: " << f1_sum / opt.outer_folds << '\n'
               << "Total selection time (all outer folds): " << selection_time_sum << " s\n"
               << "Total final mining time: " << final_mining_sum << " s\n"
               << "Total final classification time: " << final_classification_sum << " s\n"
               << "Total dataset execution time: " << dataset_seconds << " s\n"
               << "Selected variants:\n";
        for (const auto& [name, n] : selected_counts) report << "  " << name << ": " << n << '\n';
        report << "\nGenerated logs:\n"
               << "  instance_log.csv: one row per evaluated instance (inner and outer).\n"
               << "  fold_variant_summary.csv: one row per variant/run.\n"
               << "  selection_log.csv: ranking of the structural-variant/lambda pairs actually evaluated in every outer fold.\n"
               << "  pn_guard.csv: training-only decisions to evaluate or skip PN variants.\n"
               << "  cns_guard.csv: structural and bounded-pilot CNS feasibility decisions.\n"
               << "  run_configuration.txt: complete reproducibility configuration.\n"
               << "  dataset_summary.csv: one row per outer fold with the selected variant and detailed counts.\n"
               << "  AnalisisInstancias.dat: controlled by --analysis-scope (default: final).\n"
               << "  instance_log.csv: controlled by --instance-log-scope (default: final).\n";

        std::cout << "Mean outer accuracy=" << std::fixed << std::setprecision(4) << acc_sum/opt.outer_folds
                  << " mean outer macroF1=" << f1_sum/opt.outer_folds << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Auto-DMC-AC error: " << e.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
