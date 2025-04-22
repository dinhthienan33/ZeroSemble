import json
import sys, os



res_file = "results.json" # The path of the prediction file (results.json)
ref_file = "reference.json" # The path of the Ground True file (reference.json)
output_dir = "source" #The path of the output directory
scores_file = os.path.join(output_dir, "scores.json")#The path of the output file (scores.json)





# EI counter
EI_tp = 0
EI_gold_len = 0
EI_pred_len = 0

# EC counter
EC_tp = 0
EC_gold_len = 0
EC_pred_len = 0

# RE counter
RE_GEN_tp = 0
RE_STRICT_tp = 0
RE_gold_len = 0
RE_pre_len = 0

cnt = 0



# res_file = "D:/codabench/res.json"
# ref_file = "D:/codabench/DocIE_curr copy/reference_data/reference.json"

# result_path = "/home/chenzhb/Workspaces/results.json"
# Ground_True_path = "/home/chenzhb/Workspaces/GroundTrue.json"


#################### Methods ####################
def traverse_dir(path):
    for root, dirs, files in os.walk(path):
        print("Current Directory", root)
        print("Parents：", dirs)
        print("File List：", files)

def get_mention_list(head_entity, mention_for_re):
    for mtlist in mention_for_re:
        if head_entity in mtlist:
            return mtlist
    return None


def safe_div(a, b):
    if b == 0.0:
        return 0.0
    else:
        return round(a / b * 100, 2)


def safe_div_(a, b):
    if b == 0.0:
        return 0.0
    else:
        return round(a / b, 2)


def compute_f1(cnt, tp, pred_num, gold_num):
    result = {}
    result["总样本数"] = cnt
    result["P"] = safe_div(tp, pred_num)
    result["R"] = safe_div(tp, gold_num)
    result["F1"] = safe_div_(2 * result["P"] * result["R"], result["P"] + result["R"])
    print(result)
    return result


########### Processing Ground True ##############
#traverse_dir("../")
# Lodading Ground True
with open(ref_file, "r", encoding="utf-8") as gt_file:
    # import pdb;pdb.set_trace()
    Ground_True = json.load(
        gt_file
    )  # Ground_True[0]['id', 'domain', 'title', 'doc', 'entities', 'triples', 'label_set', 'entity_label_set']


GT = {}
for doc_id, sample in Ground_True.items():

    mention_gt = sample["entities"]
    mentions_gt_list = []
    metion_type_list = []
    for i in range(len(mention_gt)):
        mention_gt[i]["mentions"] = set(
            mention_gt[i]["mentions"]
        )  # convert to set, de-dupulicate
        mentions_gt_list.append(mention_gt[i]["mentions"])
        metion_type_list.append(mention_gt[i]["type"])

    triple_gt_list = []
    triple_gt = sample["triples"]
    for gt in triple_gt:
        # import pdb;pdb.set_trace()
        triple_gt_list.append((gt["head"], gt["relation"], gt["tail"]))

    GT[doc_id] = {
        "mentions_GT": mentions_gt_list,
        "relations_GT": triple_gt_list,
        "mention_type": metion_type_list,
    }
del Ground_True
####################################################
# Loading result
with open(res_file, "r", encoding="utf-8") as f:
    results = json.load(f)
    cnt = len(results)

for pre_id, sample in results.items():
    # For each predited sample
    mention_gt_list = GT[pre_id]["mentions_GT"]
    relation_gt_list = GT[pre_id]["relations_GT"]
    type_gt_list = GT[pre_id]["mention_type"]

    mention_for_re = []
    # ================  mentions ================
    mention_pred = sample["entities"]
    for i in range(len(mention_pred)):
        mention_pred[i]["mentions"] = set(
            mention_pred[i]["mentions"]
        )  # convert to set, de-dupulicate

    # for i in range(len(mention_gt)):
    #     mention_gt[i]["mentions"] = set(mention_gt[i]["mentions"])  # convert to set, de-dupulicate
    #     mention_for_re.append(mention_gt[i]["mentions"])

    EI_gold_len += len(mention_gt_list)
    EC_gold_len += len(mention_gt_list)

    for i in range(len(mention_pred)):
        EI_pred_len += 1
        EC_pred_len += 1

        for j in range(len(mention_gt_list)):
            if (
                mention_pred[i]["mentions"] in mention_gt_list
            ): 
                EI_tp += 1
                type_idx = mention_gt_list.index(
                    mention_pred[i]["mentions"]
                ) 

                if mention_pred[i]["type"] == type_gt_list[type_idx]:
                    EC_tp += 1
                    break
                else:
                    break

    # ================  relations  ================
    triple_pred = sample["triples"]
    RE_pre_len += len(triple_pred)
    RE_gold_len += len(relation_gt_list)

    for pred in triple_pred:
        # import pdb;pdb.set_trace()
        pred_triple = (pred["head"], pred["relation"], pred["tail"])
        if (
            pred_triple in relation_gt_list
        ):  # Determine whether the predicted triple is in GT. If not, determine whether it meets the general mode.
            RE_GEN_tp += 1
            RE_STRICT_tp += 1
        else:
            head_mention_list = get_mention_list(pred["head"], mention_gt_list)
            tail_mention_list = get_mention_list(pred["tail"], mention_gt_list)
            if head_mention_list is not None and tail_mention_list is not None:
                for head_mention in head_mention_list:
                    for tail_mention in tail_mention_list:
                        if (
                            head_mention,
                            pred["relation"],
                            tail_mention,
                        ) in relation_gt_list:
                            RE_GEN_tp += 1


print("#" * 20, "Entity Identification", "#" * 20)
entity_identification_res = compute_f1(cnt, EI_tp, EI_pred_len, EI_gold_len)
print("Correct Mention list:", EI_tp)
print("Predict Mentions:", EI_pred_len)
print("GT Mentions:", EI_gold_len)

print("#" * 20, "Entity Classification", "#" * 20)
entity_classification_res = compute_f1(cnt, EC_tp, EC_pred_len, EC_gold_len)
print(
    "Correct Mention Classfication:", EC_tp
)  
print("Predict Mentions Classfication:", EC_pred_len)
print("GT Mentions Classfication:", EC_gold_len)

print("#" * 20, "RE General Mode", "#" * 22)
re_general_res = compute_f1(cnt, RE_GEN_tp, RE_pre_len, RE_gold_len)
print(
    "Correct Relations:", RE_GEN_tp
)  
print("Predict Relations:", RE_pre_len)
print("GT Relations:", RE_gold_len)


print("#" * 20, "RE Strict Mode", "#" * 22)
re_strict_res = compute_f1(cnt, RE_STRICT_tp, RE_pre_len, RE_gold_len)
print("Correct Relations:", RE_STRICT_tp)
print("Predict Relations:", RE_pre_len)
print("GT Relations:", RE_gold_len)

with open(scores_file, "w", encoding="utf-8") as f:
    json.dump(
        {
            "entity_ident": entity_identification_res["F1"],
            "entity_cla": entity_classification_res["F1"],
            "re_general": re_general_res["F1"],
            "re_strict": re_strict_res["F1"],
        },
        f,
        ensure_ascii=False,
        indent=4,
    )