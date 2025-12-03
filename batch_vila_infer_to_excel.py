import os
import pandas as pd
from vila.inference import load_model_and_tokenizer, infer_image

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_FOLDER = "inference_images"        # your folder
OUTPUT_EXCEL = "inference_output.xlsx"   # output file

MODELS = {
    "NVILA-15B": "Efficient-Large-Model/NVILA-15B",
    "NVILA-8B":  "Efficient-Large-Model/NVILA-8B"
}

PROMPT = "Please describe the image"
# -------------------------------

def run_inference_on_model(model_name, model_path, image_list):
    print(f"\n🔵 Loading model: {model_name} ...")

    model, tokenizer, image_processor, conv_template = load_model_and_tokenizer(
        model_path=model_path,
        conv_mode="auto"
    )

    results = []
    for img in image_list:
        img_path = os.path.join(IMAGE_FOLDER, img)
        print(f"🔹 Running inference on: {img_path}")

        text_output = infer_image(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            conv_template=conv_template,
            image_path=img_path,
            text=PROMPT
        )

        results.append({
            "image_name": img,
            "model": model_name,
            "description": text_output
        })

    return results


def main():
    images = [f for f in os.listdir(IMAGE_FOLDER)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(images) == 0:
        print("❌ No images found in the inference_images/ folder.")
        return

    final_results = []

    # Run inference for each model
    for model_name, model_path in MODELS.items():
        model_results = run_inference_on_model(model_name, model_path, images)
        final_results.extend(model_results)

    # Save to Excel
    df = pd.DataFrame(final_results)
    df.to_excel(OUTPUT_EXCEL, index=False)

    print(f"\n✅ DONE! Results saved to: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
