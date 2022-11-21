from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect

from PIL import Image
import numpy as np
import pytesseract

from transformers import BertTokenizerFast, EncoderDecoderModel

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ivan.ustinov\Tesseract-OCR\tesseract.exe'

tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
model = EncoderDecoderModel.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')


def home(request):
    return render(request, "main/home.html")


def convert2text(request):

    if request.method == 'POST' and request.FILES.get('file_for_conversion') is not None:
        image = request.FILES['file_for_conversion'].file
        image = Image.open(image)
        text_res = pytesseract.image_to_string(image)

        print(text_res)

        request.session["text_result"] = text_res
        return HttpResponseRedirect("/convert2text")

    return render(request, "main/convert2text.html")


def generate_summary(text):
    # cut off at BERT max length 512
    inputs = tokenizer([text], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    output = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.decode(output[0], skip_special_tokens=True)


def summarization(request):

    if request.method == 'POST':
        text = request.POST.get('text')
        text_res = generate_summary(text)

        request.session["text_summarization"] = text_res
        return HttpResponseRedirect("/summarization")

    return render(request, "main/summarization.html")

