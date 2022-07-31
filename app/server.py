from copyreg import pickle
import aiohttp
import asyncio
import uvicorn
import os
# import fastbook
# from fastbook import *
from fastai import *
from fastai.imports import *
from fastai.vision import *
from fastai.vision.all import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles


import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

export_file_url = '1BGjvik7txqneiwTG9VIYcTtS9pxC9EKC'
export_file_name = 'THFood_20_epochs.pkl'

classes = ['Chicken Green Curry','Pork Curry with Morning Glory','Spicy Mixed Vegetable Soup','Pork Chopped Tofu Soup','Stuffed Bitter Gourd Broth','Chicken Mussaman Curry','Sour Soup','Stir Fried Chicken with Chestnuts','Omelette','Fried Egg',
           'Egg and Pork in Sweet Brown Sauce','Egg with Tamarind Sauce','Banana in Coconut Milk','Stir Fried Rice Noodles with Chicken','Fried Cabbage with Fish Sauce','Grilled River Prawn','Baked Prawns with Vermicelli','Coconut Rice Pancake','Mango Sticky Rice','Thai Pork Leg Stew',
          'Shrimp Paste Fried Rice','Curried Noodle Soup with Chicken','Fried Rice','Shrimp Fried Rice','Steamed Capon in Flavored Rice','Thai Chicken Biryani','Thai Chicken Coconut Soup','River Prawn Spicy Soup','Fried Fish-Paste Balls','Deep Fried Spring Roll',
          'Stir-Fried Chinese Morning Glory','Fried Noodle Thai Style with Prawn','Stir Fried Thai Basil with Minced Pork','Fried Noodle in Soy Sauce','Stir-Fried Pumpkin with Eggs','Stir-Fried Eggplant with Soybean Paste Sauce','Stir Fried Clams with Roasted Chili Paste','Golden Egg Yolk Threads','Chicken Panang','Thai Wing Beans Salad',
          'Spicy Glass Noodle Salad','Spicy Minced Pork Salad','Egg Custard in Pumpkin','Tapioca Balls with Pork Filling','Green Papaya Salad','Thai-Style Grilled Porks Skewers','Pork Satay with Peanut Sauce','Steamed Fish with Curry Paste', 'Stir Fried Thai Basil with Minced Pork with Fried Egg on top']
path = Path(__file__).resolve().parent

model_paths = os.path.join(path, export_file_name)

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


# async def download_file(url, dest):
#     if dest.exists(): return
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             data = await response.read()
#             with open(dest, 'wb') as f:
#                 f.write(data)


async def setup_learner():
    # download_file_from_google_drive(export_file_url, path / export_file_name)
    try:
        learn = load_learner(model_paths)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = PILImage.create(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
