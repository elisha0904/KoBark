# -*- coding: utf-8 -*- s
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
import json
import re

api_key = '' # api를 기입해주세요

template = '''
You are an editor who turns a typical book written in Korean into a screenplay.

I give you the book, and you do the following.

1. Infer each character's gender as either 남 or 여, their age as 아동, 성인, or 노년, and their personality as '밝은', '차분한', or '어두운' from the book.
2. For each sentence, distinguish whether it is narration or dialogue from a specific character. 
3. For each piece of character dialogue, dictate the emotion using one of the following: '기쁨', '슬픔', '분노', '불안', '상처', '당황', or '중립'. Keep the narration tone as '중립'

After completing all three steps, provide the results as follows.
The results should be presented in the form of a dictionary with two key-value pairs. One key should contain inferred character attributes, and the other key should contain the scenario. The scenario must be in the same sequential order as the sentences in the original book.

{{'Characters': {{'Person1': [gender, age range, personality], 'Person2': [gender, age range, personality],
'Scenario': [['Narrator', 'Narration Content', '중립'], ['Person1', 'Dialog Content', 'Emotion Expression'], ['Person2', 'Dialog Content', 'Emotion Expression'],
['Narrator', 'Narration Content', '중립'], ['Person1', 'Dialog Content', 'Emotion Expression']]}}

Here's an example.

input_content = '자기를 불러 멈춘 사람이 그 학교 학생인 줄 김첨지는 한번 보고 짐작할 수 있었다. 그 학생은 다짜고짜로, “남대문 정거장까지 얼마요?” 라고 물었다. 아마도 그 학교 기숙사에 있는 이로 동기 방학을 이용하여 귀향하려 함이로다. 오늘 가기로 작정은 하였건만, 비는 오고 짐은 있고 해서 어찌 할 줄 모르다가 마침 김첨지를 보고 뛰어나왔음이리라. 그렇지 않다면 왜 구두를 채 신지 못해서 질질 끌고, 비록 ‘고꾸라’ 양복일망정 노박이로 비를 맞으며 김첨지를 뒤쫓아 나왔으랴. “남대문 정거장까지 말씀입니까?” 하고, 김첨지는 잠깐 주저하였다.'

When these books come to you, you will do all the three things I have instructed you to do, and then give me the following answer.

{{'Characters' : {{'학생':['남', '아동', '차분한'], '김첨지': ['남', '성인', '어두운'],
'Scenario': [['Narrator', '자기를 불러 멈춘 사람이 그 학교 학생인 줄 김첨지는 한번 보고 짐작할 수 있었다.', '중립'], ['학생', '남대문 정거장까지 얼마요?', '중립'],
['Narrator', '라고 물었다. 아마도 가기로 작정은 하였건만, 비는 오고 짐은 있고 해서 어찌 할 줄 모르다가 마침 김첨지를 보고 뛰어나왔음이리라. 그렇지 않다면 왜 구두를 채 신지 못해서 질질 끌고, 비록 '고꾸라' 양복일망정 노박이로 비를 맞으며 김첨지를 뒤쫓아 나왔으랴.', '중립'],
['김첨지', '남대문 정거장까지 말씀입니까?', '당황'], ['Narrator', '하고, 김첨지는 잠깐 주저하였다.', '중립']]}}

Now I'll give you the book. When processing segments of the same book over multiple calls, remember the previous content and allocate accordingly.
Please return only in the form of 'return'. 

'''

def splitBook(book):
    sentences = re.split(r'(?<=[.!?])\s+|(?<=")\s+|(?<=\')\s+', book)
    
    chunk = ""
    text = []

    for sentence in sentences:
        if len(chunk) + len(sentence) > 4000:
            text.append(chunk.strip())
            chunk = sentence
        else:
            chunk += ' ' + sentence

    if chunk:
        text.append(chunk.strip())

    return text

def langchain(book):
    gpt4 = ChatOpenAI(model_name='gpt-4', temperature=0, openai_api_key= api_key )
    memory = ConversationSummaryMemory(llm=gpt4)

    prompt = PromptTemplate(
        input_variables=['history', 'input'],
        template= template + "The previous book's content is {history}. The current book's content is {input}. Follow the three steps I told you and give me the answer in the form of the return I gave you."
    )

    chain = ConversationChain(llm=gpt4, memory = memory, prompt=prompt)

    book_content = splitBook(book)
    book_content = [f"'''{string}'''" for string in book_content]

    result = {'Characters':{}, 'Scenario':[]}

    for content in book_content:
        content = content.replace("'",'"')
        flag = True

        while flag:
            res = chain.run(content)
            res = res.replace('"','$').replace("'", '"').replace('$',"'")
            start_index = res.find('{')
            end_index = res.rfind('}')
            res = res[start_index:end_index+1]

            try:
                res_dict = json.loads(res)
                flag = False
            except Exception as ex:
                print(ex)
                print('Try again')
                continue

        result['Scenario'].extend(res_dict['Scenario'])
        result['Characters'].update(res_dict['Characters'])

    return result