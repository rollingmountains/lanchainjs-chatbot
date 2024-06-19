import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import { RunnableWithMessageHistory } from '@langchain/core/runnables';

import * as dotenv from 'dotenv';
dotenv.config();

//Create prompt

const createPrompt = () => {
  //Create prompt
  const prompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      `You are a helpful assistant who remembers all the things user shares with you.`,
    ],
    ['placeholder', '{chat_history}'],
    ['user', '{input}'],
  ]);

  return prompt;
};

//Create chain

const createChain = (prompt) => {
  //Instantiate the model
  const model = new ChatOpenAI({
    modelName: 'gpt-3.5-turbo',
    temperature: 0.7,
  });

  //Set message history to an empty object
  const messageHistories = {};

  //Create a chain
  const chain = prompt.pipe(model);

  //Create runnable sequence
  const withMessageHistory = new RunnableWithMessageHistory({
    runnable: chain,
    getMessageHistory: async (sessionId) => {
      if (messageHistories[sessionId] === undefined) {
        messageHistories[sessionId] = new InMemoryChatMessageHistory();
      }
      return messageHistories[sessionId];
    },
    inputMessagesKey: 'input',
    historyMessagesKey: 'chat_history',
  });

  return withMessageHistory;
};

//Create chat history session id config
export const config = {
  configurable: {
    sessionId: 'abc2',
  },
};

export const initializeLangChain = async () => {
  return createChain(createPrompt());
};
// console.log(createPrompt());
// console.log(createChain(createPrompt()));

// const response = await withMessageHistory.invoke(
//   {
//     input: 'My name is Bob.',
//   },
//   config
// );

// const followupResponse = await withMessageHistory.invoke(
//   {
//     input: 'What is my name and how old am I?',
//   },
//   config
// );
// console.log(followupResponse.content);

// console.log(
//   await model.invoke([
//     new HumanMessage({ content: 'Hi My name is bob.' }),
//     new HumanMessage({ content: 'What is my name?' }),
//   ])
// );
// console.log('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@');
// console.log(
//   await model.invoke([new HumanMessage({ content: 'What is my name?' })])
// );
