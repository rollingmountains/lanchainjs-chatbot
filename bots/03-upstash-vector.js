import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import {
  RunnableSequence,
  RunnablePassthrough,
} from '@langchain/core/runnables';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
// import { EPubLoader } from '@langchain/community/document_loaders/fs/epub';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { EPubLoader } from 'langchain/document_loaders/fs/epub';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Document } from '@langchain/core/documents';
import { OpenAIEmbeddings } from '@langchain/openai';
import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase';
import { createClient } from '@supabase/supabase-js';
// import path from 'path';

import { RunnableWithMessageHistory } from '@langchain/core/runnables';

import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import path from 'path';

import { retriever } from '../utils/retriever.js';
import { PassThrough } from 'stream';

// Get the directory name of the current module file
const __dirname = dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const openaiApiKey = process.env.OPENAI_API_KEY;

//Data load, transform and embed

//Upload the epub document
const loader = new EPubLoader('../God TalkswithArjuna.epub', {
  splitChapters: true,
});

const docs = await loader.load();

const removeChapters = [
  'Praise for Paramahansa Yogananda’s commentary on the Bhagavad Gita…',
  'Acknowledgments',
  'Back Cover',
];

const parsedFilteredDoc = docs
  .filter(
    (doc) =>
      !removeChapters.includes(doc.metadata.chapter) &&
      doc.pageContent !== undefined &&
      doc.metadata.chapter !== undefined
  )
  .map((doc) => ({
    pageContent: doc.pageContent
      .replace(/\n/g, ' ')
      .replace(/\+/g, '')
      .replace(/\[[^\[\]]*\]/g, ''),
    metadata: doc.metadata,
  }));

// console.log('filteredDocs: ', parsedFilteredDoc[0]);

const formattedDoc = parsedFilteredDoc.map(
  (doc) =>
    new Document({
      pageContent: doc.pageContent,
      metadata: {
        source: doc.metadata.source,
        chapter: doc.metadata.chapter ? doc.metadata.chapter : undefined,
      },
    })
);

console.log('formattedDoc: ', formattedDoc);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  separators: ['\n\n', '\n', ' ', ''],
  chunkOverlap: 50,
});

const splitDoc = await splitter.splitDocuments(formattedDoc);

export function combineDocuments(docs) {
  return docs.map((doc) => doc.pageContent).join('\n\n');
}

//create standaloneprompt template
const standaloneQuestionTemplate = `Given a question, convert it into a standalone question.
question: {question} standalone question:`;

//Standalone prompt
const standaloneQuestionPrompt = ChatPromptTemplate.fromTemplate(
  standaloneQuestionTemplate
);

//Create Answer prompt
const answerTemplate = `You are an expert who can answer questions from the the book God talks with Arjuna by Yogananda Parmahansa based on the context provided. Try to find the answer in the context. You are a friendly bot. If you really don't know the answer just say 'I am sorry. I do not know the answer.' Do not make up the answer.
context: {context}
question: {question}
answer:  `;

const answerPrompt = ChatPromptTemplate.fromTemplate(answerTemplate);

function combineRetrieverDoc(docs) {
  return docs.map((doc) => doc.pageContent).join('\n');
}

// Instantiate the model
const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
  apiKey: openaiApiKey,
});

const parser = new StringOutputParser();

const standAloneQuestionChain = RunnableSequence.from([
  standaloneQuestionPrompt,
  model,
  parser,
]);

const retrieverChain = RunnableSequence.from([
  (prevResult) => prevResult.standalone_question,
  retriever,
  combineDocuments,
]);

const answerChain = RunnableSequence.from([answerPrompt, model, parser]);

const chain = RunnableSequence.from([
  {
    standalone_question: standAloneQuestionChain,
    original_input: new RunnablePassthrough(),
  },
  {
    context: retrieverChain,
    question: ({ original_input }) => original_input.question,
  },
  answerChain,
]);

const response = await chain.invoke({
  question: `At what point in time Parmahamsa arrived in USA to spread yoga?`,
});

console.log(response);
