using System;
using BlingFire;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchScriptAndBlingFireExamples
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //DistilBERTAndBlingFire.Run();
            MNIST.Run();
        }
    }
}
