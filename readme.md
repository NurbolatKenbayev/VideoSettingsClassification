# Guide to run 

**Install modules.** 
`pip install scenedetect opencv-python transformers torch asyncio aiofiles openai`

`pip install python-dotenv`

`pip install face_recognition` (requires `dlib`, which in turn requires C++ build tools. You will need to have `cmake`, `gcc` and `g++` on your system. Install `pip install dlib` before installing `face_recognition`.)
or if it doesn't work install manually (**preferred**):
(You will need to have `cmake` on your system: `sudo apt update` + `sudo apt install cmake`)
Go to https://pypi.org/project/face-recognition/#files and download `face_recognition-1.3.0.tar.gz`
Extract the downloaded package `tar -xvzf face_recognition-1.3.0.tar.gz`
Install the package:
`cd face_recognition-1.3.0`
`python3 setup.py install`
Verify installation in python: `import face_recognition`
(`pip show face_recognition` to find the location of the installed package)

`pip install networkx`

`pip install boto3`

Don't forget to add `.env` file with the `OPENAI_API_KEY` and **put it in the `.gitignore`**.


**Pipeline.**

1.	VideoProcessor: Detects scenes and outputs scene metadata.

2.	FrameExtractor: Extracts frames for each scene and updates metadata with frame paths.

3.	ImageCaptioningAgent: Generates captions for each frame and updates metadata with captions.

4.	SettingClassifierAgent: Uses the captions to classify the settings.

Run `python main.py --video_path "./input_data/minecraft.mp4" --num_processes 28 --possible_settings_path "./input_data/possible_settings_minecraft_processed.json"`

As a result there will be `scenes_with_settings_predicted.json` a list of dictionaries representing separate cut scenes. Each cut scene contains (key `captions`) a list of frames within this scene with the predicted settings.


**Visualize results.** 
`brew install ffmpeg` (installed locally, to put subtitles on the video based on provided 'subtitles.ass' file)
`ffmpeg -version` (verify installation)

Run the script 
`python generate_subtitles.py --path_to_subtitle_data "../results/scenes_with_settings_predicted.json"` 
to generate `subtitles.ass` file. 
Provide `path_to_subtitle_data` path to the corresponding labeled/predicted list of dictionaries.

Run the command below to generate `output_video.mp4` video with subtitles given by `subtitles.ass` file:
`ffmpeg -i input_video.mp4 -vf "ass=subtitles.ass" -c:a copy output_video.mp4`


**Metrics.**
Since the domain of possible settings is not well-defined and I performed handmade labeling I assume it to be not perferct. Therefore, I chose Cohen’s Kappa to evaluate the level of agreement between my labeling and model's predictions.



# Notes

### Understanding OpenAI API Behavior

**Independent API Calls.** 
Each call to `openai.ChatCompletion.create()` is stateless and independent unless you include conversation history explicitly. The API does not maintain conversation state between requests unless you provide the conversation history in the messages parameter. Each API call is self-contained based on the messages you provide.

**Thread-Safety and Process-Safety.** 
The OpenAI API and its Python client library are designed to be thread-safe and can be used across multiple threads and processes. Using the same API key in multiple processes is acceptable. The API key identifies your account and is used for authentication and billing. Since each process provides its own messages, there is no risk of cross-talk or interference between processes. You need to be cautious about exceeding the API’s rate limits when making concurrent requests.



### Speed

**Multiprocessing.**
In Python, when using `multiprocessing.Pool`, the functions (and any objects they reference) that are executed in separate processes must be **picklable**. This is because the `multiprocessing` module serializes (pickles) the function and its arguments to send them to worker processes.

Methods bound to class instances cannot be pickled unless the class is defined at the top level of a module and certain conditions are met. This limitation arises because bound methods carry a reference to the instance (self), which may not be picklable.

Thus, all functions to be used in `multiprocessing.Pool` should be defined at the module level outside of the class. This makes them easily picklable and avoids issues with serialization. Also, avoid Lambdas and Nested Functions, since Lambdas and functions defined inside other functions or methods are not picklable. Stick to module-level functions. Design your multiprocessing functions to be stateless (regarding to class) or to receive all necessary data through arguments.


**Empirical observations.**
Two bottlenecks regarding time complexity are `ImageCaptioningAgent` and `SettingClassifierAgent`, where each frame processing requires a call to openai API. Thus, the multiprocessing is applied at these steps.
In the first implementation `ImageCaptioningAgent` was too slow, processing 107 cut scenes (one frame per each scene) sequentially took ~ 9 mins. But with multiprocessing over multiple cut scenes with 28 processes it took only ~ 20 secs. 