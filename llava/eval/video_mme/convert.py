import json

jinfo = json.load(open("/lustre/fs5/portfolios/nvr/projects/nvr_elm_llm/dataset/Video-MME/qa.json"))

new_jinfo = {}
for idx, obj in enumerate(jinfo):
    # print(obj)
    # print(obj.keys())

    vid = obj["video_id"]

    if vid not in new_jinfo:
        new_jinfo[vid] = {
            "video_id": obj["video_id"],
            "duration_category": obj["duration"],
            "video_category": obj["domain"],
            "video_subcategory": obj["sub_category"],
            "url": obj["url"],
            "questions": [],
        }

    new_jinfo[vid]["questions"].append(
        {
            "question_id": obj["question_id"],
            "task_type": obj["task_type"],
            "question": obj["question"],
            "choices": obj["options"],
            "answer": obj["answer"],
        }
    )
    # if idx > 5:
    #     break

new_jinfo = list(new_jinfo.values())
print(len(new_jinfo))
input()
print(json.dumps(new_jinfo, indent=2))

with open(
    "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/Video-MME/qa_old_format.json",
    "w",
) as f:
    json.dump(new_jinfo, f, indent=2)
