import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';

import * as dotenv from 'dotenv';
dotenv.config({ path: '../.env' });

//create model
const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temprature: 0.7,
});
const systemTemplate = `Translate the following into {language}`;

const prompt = ChatPromptTemplate.fromMessages([
  ['system', systemTemplate],
  ['user', '{input}'],
]);

// const result = await prompt.invoke({ language: 'Italian', input: 'Morning' });
const parser = new StringOutputParser();

await prompt.pipe(model).pipe(parser).invoke({
  language: 'Italian',
  input: 'Afternoon',
});

// await chain.invoke({
//   language: 'Italian',
//   input: 'Morning',
// });

// const message = [
//   new SystemMessage('Translate the following from English to Nepali'),
//   new HumanMessage('Hello'),
// ];

// const parser = new StringOutputParser();

// //create chain
// const chain = model.pipe(parser);

// //invoke chain
// await chain.invoke(message);

// export const langchain = async() => {
//   await model.invoke(message);
// }
