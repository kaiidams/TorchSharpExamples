using System;
using BlingFire;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchScriptAndBlingFireExamples
{
    /// <summary>
    /// Sentiment analysis with HuggingFace's DistilBERT.
    /// </summary>
    internal class DistilBERTAndBlingFire
    {
        public static void Run()
        {
            // Load the tokenizer
            var h = BlingFireUtils.LoadModel("./bert_base_tok.bin");

            // Load the model
            var model = torch.jit.load<Tensor, Tensor, Tensor>("./distilbert-base-uncased-finetuned-sst-2-english.pt");

            string[] inputs = new string[] {
                "I like you.", "You hate me?", "I like you. I love you" 
            };
            int max_len = 128;
            var input_ids = torch.zeros(inputs.Length, max_len, dtype: torch.int64);
            var attention_mask = torch.zeros(inputs.Length, max_len, dtype: torch.int64);
            for (int i = 0; i < inputs.Length; i++)
            {
                string input = inputs[i];
                byte[] inBytes = System.Text.Encoding.UTF8.GetBytes(input);
                int[] Ids = new int[max_len - 2]; // -2 for For CLS and SEP.
                int outputCount = BlingFireUtils.TextToIds(h, inBytes, inBytes.Length, Ids, Ids.Length, 0);
                Console.WriteLine(String.Format("return length: {0}", outputCount));
                input_ids[i, TensorIndex.Slice(1, -1)] = torch.from_array(Ids);
                input_ids[i, 0] = 101; // CLS
                input_ids[i, outputCount] = 102; // SEP
                attention_mask[i, (0..(outputCount + 2))] = 1;
            }

            using (torch.no_grad())
            {
                var logits = model.forward(input_ids, attention_mask);
                var pred = torch.nn.functional.softmax(logits, -1);

                for (int i = 0; i < inputs.Length; i++)
                {
                    Console.WriteLine("{0}: {1} {2}", i, pred[i, 1].item<float>(), inputs[i]);
                }
            }
        }
    }
}
