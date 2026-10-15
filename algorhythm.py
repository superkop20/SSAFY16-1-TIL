## AI2 역량평가 대비 import 명단정리
# 1-1
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# 1-2
import torch
# → nn = Neural Network의 줄임말. 모델(레이어)을 만드는 도구가 들어있습니다.
import torch.nn as nn
# → optim = Optimizer의 줄임말. 학습 알고리즘이 들어있습니다.
import torch.optim as optim
import random
from torch.utils.data import DataLoader
# 2-1
import torch.nn.functional as F
from pathlib import Path
from urllib.request import Request, urlopen
import os
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# 2-2
from dotenv import load_dotenv
from os import getenv
from openai import OpenAI
import time
import json
import httpx
import asyncio
from pprint import pprint
# 3-1
import torchvision
# torchvision 내의 transforms는 이미지 전처리(augmentation, normalization 등) 기능을 제공합니다.
import torchvision.transforms as transforms
# 3-2
# 'diffusers' 라이브러리에서 이미지 생성 파이프라인을 가져옵니다. 
from diffusers import StableDiffusionPipeline
# PIL(Pillow) 라이브러리에서 이미지 처리를 위한 Image 모듈을 가져옵니다.
from PIL import Image
# 'transformers' 라이브러리에서 CLIP 모델과 전처리기(Processor)를 가져옵니다.
from transformers import CLIPProcessor, CLIPModel
from torchvision import models
from torchvision.datasets import ImageFolder
# 4-1
import warnings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
# 4-2
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import re
from typing import List
# 5-1
from unsloth import FastModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only
import json as _json
# 5-2
from transformers import AutoModelForCausalLM
import logging
import gc
