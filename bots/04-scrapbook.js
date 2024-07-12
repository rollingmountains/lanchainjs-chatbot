/*
Input: i dont liked mondays
Punctuation: I don't liked Mondays.
Grammar: I don't like Mondays.
Translation: Italian

*/

//Import dependencies
import { ChatOpenAI } from '@langchain/openai';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { PromptTemplate } from '@langchain/core/prompts';
// import { RunnableSequence } from 'langchain/schema/runnable';
import {
  RunnableSequence,
  RunnablePassthrough,
} from '@langchain/core/runnables';

//Import .env file
import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import path from 'path';

// Get the directory name of the current module file
const __dirname = dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, '../.env') });

//Instantiate model
const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
});

const punctuationTemplate = `Given a sentence, please add only punctuation where needed.
sentence: {input}
sentence with punctuation: `;

const punctuationPrompt = PromptTemplate.fromTemplate(punctuationTemplate);

const grammarTemplate = `Given a sentence, please correct only grammar where needed.
sentence: {punctuated_sentence}
sentence with grammar:`;

const grammarPrompt = PromptTemplate.fromTemplate(grammarTemplate);

const translationTemplate = `Given a sentence, please translate into {language}
sentence: {grammatically_correct_sentence}
translated sentence: `;

const translationPrompt = PromptTemplate.fromTemplate(translationTemplate);

const punctuationChain = RunnableSequence.from([
  punctuationPrompt,
  model,
  new StringOutputParser(),
]);

const grammarChain = RunnableSequence.from([
  grammarPrompt,
  model,
  new StringOutputParser(),
]);

const translationChain = RunnableSequence.from([
  translationPrompt,
  model,
  new StringOutputParser(),
]);

const chain = RunnableSequence.from([
  {
    punctuated_sentence: punctuationChain,
    original_input: new RunnablePassthrough(),
  },
  {
    grammatically_correct_sentence: grammarChain,
    language: ({ original_input }) => original_input.language,
  },
  translationChain,
]);

const response = await chain.invoke({
  input: 'i dont liked mondays',
  language: 'french',
});

console.log(response);

export { splitDocs };
