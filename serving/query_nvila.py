from PIL import Image
from openai import OpenAI
from io import BytesIO
import base64

client = OpenAI(
    base_url="http://localhost:8000",
    api_key="fake-key",
)
# Efficient-Large-Model/nvila-8b-dev

def file_to_base64_binary(file_path: str):
    # assert file_path.lower().endswith(".mp4", ".jpg", ".jpeg", ".png", )
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def file2base(image_url):
    with open(image_url, "rb") as image_file:
        # Convert the image to PNG and save it as a temporary file
        image = Image.open(image_url)
        temp_file = BytesIO()
        image.save(temp_file, format="PNG")
        temp_file.seek(0)
        base64_image = base64.b64encode(temp_file.read()).decode("utf-8")
    return base64_image

def main(model: str = "NVILA-8B",  stream: bool = True):
    image_url = "inference_test/test_data/caption_meat.jpeg"
    video_url = "https://avtshare01.rz.tu-ilmenau.de/avt-vqdb-uhd-1/test_1/segments/bigbuck_bunny_8bit_200kbps_360p_60.0fps_hevc.mp4"
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            # "url": video_url,
                            # Or you can pass in a base64 encoded image
                            "url": f"data:video/mp4;base64,{file_to_base64_binary('./serving/bigbuck_bunny_8bit_200kbps_360p_60.0fps_hevc.mp4')}",
                        },
                        "frames": 16,
                    },
                    {"type": "text", "text": "Please describe the video"},
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": "https://blog.logomyway.com/wp-content/uploads/2022/01/NVIDIA-logo.jpg",
                    #         # Or you can pass in a base64 encoded image
                    #         # "url": f"data:image/png;base64,{base64_image}",
                    #     },
                    # },
                ],
            }
        ],
        model=model,
        stream=stream,
        # NOTE(ligeng): NVILA current does not support these parameters
        # max_tokens=300,
        # You can pass in extra parameters as follows
        # extra_body={"num_beams": 1, "use_cache": True},
    )

    if stream:
        idx = 0
        for chunk in response:
            print(chunk.choices[0].delta.content, end="")
            idx += 1
        print()
    else:
        print(response.choices[0].message.content[0]["text"])
        
if __name__ == "__main__":
    import fire
    fire.Fire(main)
