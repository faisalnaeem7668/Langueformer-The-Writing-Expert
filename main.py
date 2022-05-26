import streamlit as st
from styleformer import Styleformer
import numpy as np
import json

from annotated_text import annotated_text
from bs4 import BeautifulSoup
from gramformer import Gramformer
import pandas as pd
import torch
import math
import re


st.set_page_config(
    page_title="Langueformer App",
    initial_sidebar_state="expanded"
)

print('Wait...Model is loading!!!')

class Style_Demo:
    def __init__(self):

        self.style_map = {
            'ctf': ('Casual to Formal', 0),
            'ftc': ('Formal to Casual', 1),
            'atp': ('Active to Passive', 2),
            'pta': ('Passive to Active', 3)
        }
        self.inference_map = {
           -1: 'Regular model on CPU',
            1: 'Regular model on GPU',
            2: 'Quantized model on CPU'
        }
        with open("streamlit_examples.json") as f:
            self.examples = json.load(f)

    @st.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
    def load_sf(self, style=0):
        sf = Styleformer(style=style)
        return sf

    def main(self):
        st.title("Text Transformer")
        st.write('A Neural Language Style Transfer framework to transfer natural language text smoothly between fine-grained language styles like formal/casual, active/passive, and many more')

        style_key = st.sidebar.selectbox(
            label='Transformation Style',
            options=list(self.style_map.keys()),
            format_func=lambda x: self.style_map[x][0]
        )
        exp = st.sidebar.expander('Settings', expanded=True)
        with exp:
            inference_on = exp.selectbox(
                label='Inference on',
                options=list(self.inference_map.keys()),
                format_func=lambda x: self.inference_map[x]
            )
            quality_filter = exp.slider(
                label='Quality filter',
                min_value=0.5,
                max_value=0.99,
                value=0.95
            )
            max_candidates = exp.number_input(
                label='Text Polishing',
                min_value=1,
                max_value=20,
                value=5
            )
        with st.spinner('Please Wait!!! Model Is Loading...'):
            sf = self.load_sf(self.style_map[style_key][1])
        input_text = st.selectbox(
            label="Choose an example",
            options=self.examples[style_key]
        )
        input_text = st.text_input(
            label="Input your text here",
            value=input_text
        )

        if input_text.strip():
            result = sf.transfer(input_text, inference_on=inference_on,
                                 quality_filter=quality_filter, max_candidates=max_candidates)
            st.markdown(f'#### Output:')
            st.write('')
            if result:
                st.code(result)
            else:
                st.info('No good quality transformation available !')
        else:
            st.warning("Please select/enter new text to proceed")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(1212)


class GramformerDemo:

    def __init__(self):

        self.model_map = {
            'Corrector': 1
           
        }
        self.examples = [
            "what be the reason for everyone leave the comapny",
            "He are moving here.",
            "I am doing fine. How is you?",
            "Matt like fish",
            "the collection of letters was original used by the ancient Romans", 
            "I walk to the store and I bought milk",
           
        ]

    @st.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
    def load_gf(self, model: int):
        """
            Load Grammar corrector/highlighter model
        """
        gf = Gramformer(models=model, use_gpu=False)
        return gf

    def show_highlights(self, gf: object, input_text: str, corrected_sentence: str):
        """
            To show highlights
        """
        try:
            def strikeout(x): return '\u0336'.join(x) + '\u0336'
            highlight_text = gf.highlight(input_text, corrected_sentence)
            color_map = {'d': '#faa', 'a': '#afa', 'c': '#fea'}
            tokens = re.split(r'(<[dac]\s.*?<\/[dac]>)', highlight_text)
            annotations = []
            for token in tokens:
                soup = BeautifulSoup(token, 'html.parser')
                tags = soup.findAll()
                if tags:
                    _tag = tags[0].name
                    _type = tags[0]['type']
                    _text = tags[0]['edit']
                    _color = color_map[_tag]

                    if _tag == 'd':
                        _text = strikeout(tags[0].text)

                    annotations.append((_text, _type, _color))
                else:
                    annotations.append(token)
            args = {
                'height': 45*(math.ceil(len(highlight_text)/100)),
                'scrolling': True
            }
            annotated_text(*annotations)
        except Exception as e:
            st.error('Some Error Occured!')
            print(e)
            st.stop()

    def show_edits(self, gf: object, input_text: str, corrected_sentence: str):
        """
            To show edits
        """
        try:
            edits = gf.get_edits(input_text, corrected_sentence)
            df = pd.DataFrame(edits, columns=['type', 'Original Word', 'Original Start',
                              'Original End', 'Correct Word', 'Correct Start', 'Correct End'])
            df = df.set_index('type')
            st.table(df)
        except Exception as e:
            st.error('Some Error Occured!')
            print(e)
            st.stop()

    def main(self):
        st.title("Grammar Corrector/Highlighter")

        model_type = st.sidebar.selectbox(
            label='Choose Model',
            options=list(self.model_map.keys())
        )
        if model_type == 'Corrector':
            max_candidates = st.sidebar.number_input(
                label='Max candidates',
                min_value=1,
                max_value=10,
                value=1
            )
        
        with st.spinner('Please Wait!!! Model Is Loading...'):
            gf = self.load_gf(self.model_map[model_type])

        input_text = st.selectbox(
            label="Choose an example",
            options=self.examples
        )
        input_text = st.text_input(
            label="Input your text here",
            value=input_text
        )

        if input_text.strip():
            results = gf.correct(input_text, max_candidates=max_candidates)
            corrected_sentence = list(results)[0]
            st.markdown(f'#### Output:')
            st.write('')
            st.code(corrected_sentence)

            exp1 = st.expander(label='Highlights', expanded=True)
            with exp1:
                self.show_highlights(gf, input_text, corrected_sentence)

            exp2 = st.expander(label='Corrections',expanded=True)
            with exp2:
                self.show_edits(gf, input_text, corrected_sentence)

        else:
            st.warning("Please select/enter new text to proceed")


sidebar = st.sidebar
sidebar.image('Capture.jpg')
options = ['About Langueformer', 'Grammar Corrector/Highlighter', 'Text Transformer']
selOpt = sidebar.selectbox('Choose option', options)


def home():
    
    st.title('Langueformer')
    st.write('Langueformer is an AI Writing Assistant that helps to improve the English language in terms of vocabulary like error detection and correction , conversion of casual text to formal and vice-versa , conversion of active voice to passive voice and vice-versa.')
    st.header('Grammar Corrector/Highlighter')
    st.write('Human and machine generated text often suffer from grammatical or typographical errors. It can be spelling, punctuation, grammatical or word choice errors. Gramformer is a model that is capable of detecting, highlighting and correcting grammatical errors. To make sure the corrections and highlights recommended are of high quality, it comes with a quality estimator.')
    st.subheader('Usecases for Grammar Corrector/Highlighter')
    st.success("1: Assisted writings for humans.")
    st.info("2: Highlighter helps in showing errors in different parts of speech.")
    st.subheader('Sample for Corrector:')
    st.image('img_gram_1.png')
    st.subheader('Sample for Highlighter:')
    st.image('img_gram_2.png')
    st.header('Text Transformer')
    st.write('Text Transformer is one of the important tool of this web app as it is capable of transforming the casual text into formal text and vice-versa. Likewise it can also convert the active speech into passive speech and vice-versa.')   
    st.subheader('Usecases for Text Transformer')
    st.success("1: Perfect tool for writings on social media platforms.")
    st.warning("2: Frequent usage of different suggestions improves the vocabulary of an individual.")
    st.error("3: Helps us to write or check mails/letters at fast pace.")
    st.image('img_style_0.png')
    st.subheader('Sample for Casual to Formal Transformation:')
    st.image('img_style_1.png')
    st.subheader('Sample for Formal to Casual Transformation:')
    st.image('img_style_2.png')
    st.subheader('Sample for Active to Passive Transformation:')
    st.image('atp.png')
    st.subheader('Sample for Passive to Active Transformation:')
    st.image('pta.png')

    
def gram():
    obj = GramformerDemo()
    obj.main()


def style():
    obj = Style_Demo()
    obj.main()


if selOpt == options[0]:
    home()
elif selOpt == options[1]:
    gram()
elif selOpt == options[2]:
    style()
