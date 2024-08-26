from transformers import pipeline
from PIL import Image
import os
import sys
import shutil
import torch
import spacy 
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from sentence_transformers import SentenceTransformer


CAPTIONER_MODEL = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
WORD_EMBEDDER = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


try:
    SENTENCE_PARSER = spacy.load('en_core_web_md')
except:
    print("Downloading en_core_web_md...")
    spacy.cli.download('en_core_web_md')
    SENTENCE_PARSER = spacy.load('en_core_web_md')

ITER_LIMIT = 16

def save_captions(image_path, save_path, files):
    ITER = 0
    ims = []
    filenames = []
    for file in tqdm(files):
        ims.append(Image.open(os.path.join(image_path, file)))
        print(file)
        filenames.append(file.split(".")[0] + ".npy")
        ITER += 1
        if ITER < ITER_LIMIT:
            continue
        else:
            im_cap = get_activities(ims)
            for i, nf in enumerate(tqdm(filenames, leave=False)):
                print(i)
                with open(os.path.join(save_path, nf), 'wb') as f:
                    np.save(f, im_cap[i,:])
            ITER = 0
            ims = []
            filenames = []
    if len(ims) != 0:
        im_cap = get_activities(ims)
        for i, nf in enumerate(filenames):
            with open(os.path.join(save_path, nf), 'wb') as f:
                np.save(f, im_cap[i,:])


def get_activities(images):
    captions = CAPTIONER_MODEL(images)
    activities = get_activities_from_captions(captions)
    embedded_activities = embed_activities(activities)
    return torch.from_numpy(embedded_activities)

def embed_activities(activities):
    embedded_activities = WORD_EMBEDDER.encode(activities)
    return embedded_activities

def get_activities_from_captions(captions):
    activities = []
    for caption in captions:
        activities.append(get_activity_from_caption(caption[0]['generated_text']))
    return activities

def get_activity_from_caption(caption):
    parsed_caption = SENTENCE_PARSER(caption)
    activities = []
    for tok in parsed_caption:
        if tok.pos_ == "VERB":
            activities.append(str(tok))
        if tok.pos_ == "NOUN" and tok.dep_ == "dobj":
            activities.append(str(tok))
    return " ".join(activities)


def translate_vasr(df):
    for col in df:
        if isinstance(df[col][0], str):
            df[col] = df[col].str.replace(".jpg", ".npy").replace(".jpeg", ".npy").replace(".png", ".npy")
    return df


def caption_bongard_hoi(dataset_path, save_path):
    files = os.listdir(dataset_path)
    files = sorted(files)
    save_captions(dataset_path, save_path, files)

def caption_bongard_hoi_part(dataset_path, save_path, n):
    files = os.listdir(dataset_path)
    files = sorted(files)
    files = files[int(n):]
    save_captions(dataset_path, save_path, files)



def caption_vasr(dataset_path, save_path):
    image_dataset_path = os.path.join(dataset_path, "images_512")
    files = os.listdir(image_dataset_path)
    files = sorted(files)
    save_path_images = os.path.join(save_path, "images_512")
    os.makedirs(save_path_images, exist_ok=True)

    save_captions(image_dataset_path, save_path_images, files)

    for file_path in glob.glob(os.path.join(dataset_path, "*.csv")):
        _, filename = os.path.split(file_path)
        annots = pd.read_csv(file_path)
        annots_translated = translate_vasr(annots)
        annots_translated.to_csv(os.path.join(save_path, filename), index=False)

def caption_vasr_part(dataset_path, save_path, n):
    image_dataset_path = os.path.join(dataset_path, "images_512")
    files = os.listdir(image_dataset_path)
    files = sorted(files)
    files = files[int(n):]
    save_path_images = os.path.join(save_path, "images_512")
    os.makedirs(save_path_images, exist_ok=True)

    save_captions(image_dataset_path, save_path_images, files)

    for file_path in glob.glob(os.path.join(dataset_path, "*.csv")):
        _, filename = os.path.split(file_path)
        annots = pd.read_csv(file_path)
        annots_translated = translate_vasr(annots)
        annots_translated.to_csv(os.path.join(save_path, filename), index=False)



if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    # args[3] = save_path
    os.makedirs(args[3], exist_ok=True)
    globals()[args[1]](*args[2:])