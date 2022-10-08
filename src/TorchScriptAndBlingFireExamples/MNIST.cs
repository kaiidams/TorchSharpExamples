using System;
using BlingFire;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchScriptAndBlingFireExamples
{
    /// <summary>
    /// This trains ScriptModel created in Python.
    /// </summary>
    internal class MNIST
    {
        public static void Run()
        {
            int batch_size = 64;
            var train_dataset = torchvision.datasets.MNIST(".", true, download: true);
#if true
            var model = torch.jit.load<Tensor, Tensor>("./mnist-mlp.pt");
#else
            var model = nn.Sequential(
                nn.Linear(28 * 28, 256),
                nn.ReLU(),
                nn.Linear(256, 10));
#endif
            var train_loader = new torch.utils.data.DataLoader(train_dataset, batch_size, true);
            model.train();
            var critera = torch.nn.CrossEntropyLoss();
            var optimizer = torch.optim.SGD(model.parameters(), 1e-2);
            for (int i = 0; i < 5; i++)
            {
                var step = 0;
                foreach (var batch in train_loader)
                {
                    optimizer.zero_grad();
                    var input = batch["data"].reshape(-1, 28 * 28);
                    var output = model.forward(input);
                    var loss = critera.forward(output, batch["label"]);
                    loss.backward();
                    optimizer.step();
                    step++;
                    if (step % 100 == 0)
                    {
                        Console.WriteLine("step: {0}, loss: {1}", step, loss.detach().cpu().item<float>());
                    }
                }
            }
        }
    }
}
