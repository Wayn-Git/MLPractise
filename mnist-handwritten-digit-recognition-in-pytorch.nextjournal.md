# MNIST Handwritten Digit Recognition in PyTorch

In this article we'll build a simple convolutional neural network in PyTorch and train it to recognize handwritten digits using the MNIST dataset. Training a *classifier* on the MNIST dataset can be regarded as the *hello world* of image recognition.

![55154820_a43d3eb317_o.jpg][nextjournal#file#cc8b4b29-c59f-4cae-a9bf-51888d7388f8]

MNIST contains 70,000 images of handwritten digits: 60,000 for training and 10,000 for testing. The images are grayscale, 28x28 pixels, and centered to reduce preprocessing and get started quicker. 

# Setting up the Environment

We will be using [PyTorch](https://pytorch.org/) to train a convolutional neural network to recognize MNIST's handwritten digits in this article. PyTorch is a very popular framework for deep learning like [Tensorflow](tensorflow.org), [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/) and [Caffe2](https://caffe2.ai/). But unlike these other frameworks PyTorch has dynamic execution graphs, meaning the computation graph is created on the fly.

Since there's already a PyTorch environment from another article, we can just transclude it and use it here.

```python id=7bc8789f-80f8-4f87-bb55-5a6ece76684d
import torch
import torchvision
```

# Preparing the Dataset

With the imports in place we can go ahead and prepare the data we'll be using. But before that we'll define the hyperparameters we'll be using for the experiment. Here the number of epochs defines how many times we'll loop over the complete training dataset, while `learning_rate` and `momentum` are hyperparameters for the optimizer we'll be using later on.

```python id=2b7f273e-ceac-4907-806f-8279409c43ba
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
```

For repeatable experiments we have to set random seeds for anything using random number generation - this means `numpy` and `random` as well! It's also worth mentioning that cuDNN uses nondeterministic algorithms which can be disabled setting `torch.backends.cudnn.enabled = False`.

Now we'll also need DataLoaders for the dataset. This is where TorchVision comes into play. It let's use load the MNIST dataset in a handy way. We'll use a `batch_size` of 64 for training and size 1000 for testing on this dataset. The values `0.1307` and `0.3081` used for the `Normalize()` transformation below are the global mean and standard deviation of the MNIST dataset, we'll take them as a given here.

TorchVision offers a lot of handy transformations, such as cropping or normalization.

```python id=1cda50d5-732e-4852-bec7-3cb50a07616a
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
```

PyTorch's `DataLoader` contain a few interesting options other than the dataset and batch size. For example we could use `num_workers > 1` to use subprocesses to asynchronously load data or using pinned RAM (via `pin_memory`) to speed up RAM to GPU transfers. But since these mostly matter when we're using a GPU we can omit them here.

Now let's take a look at some examples. We'll use the `test_loader` for this.

```python id=46669dc1-14ff-44bb-a645-dd92074e3b41
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
```

Ok let's see what one test data batch consists of.

```python id=639bcf38-f56d-446d-a319-46d5977c1e9f
example_data.shape
```

So one test data batch is a  tensor of shape: [reference][nextjournal#reference#3e383ae1-aa7c-4289-9256-3a41ca01f830]. This means we have 1000 examples of 28x28 pixels in grayscale (i.e. no rgb channels, hence the one). We can plot some of them using matplotlib.

```python id=07b8dcbb-55f8-456b-9fd0-5d49af6701dc
import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig
```

![result][nextjournal#output#07b8dcbb-55f8-456b-9fd0-5d49af6701dc#result]

Alright, those shouldn't be too hard to recognize after some training.

[signup][nextjournal#signup#81c8c1ec-526b-46c8-a643-9b13dbb4d111]

# Building the Network

Now let's go ahead and build our network. We'll use two 2-D convolutional layers followed by two fully-connected (or *linear)* layers. As activation function we'll choose [rectified linear units](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) (ReLUs in short) and as a means of regularization we'll use two dropout layers. In PyTorch a nice way to build a network is by creating a new class for the network we wish to build. Let's import a few submodules here for more readable code.

```python id=20c1cdd6-35dd-48aa-bac9-2f31a3f37366
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

```python id=aa899593-62cf-4671-a3f5-8caa180f7344
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
```

Broadly speaking we can think of the `torch.nn` layers as which contain trainable parameters while `torch.nn.functional` are purely functional. The `forward()` pass defines the way we compute our output using the given layers and functions. It would be perfectly fine to print out tensors somewhere in the forward pass for easier debugging. This comes in handy when experimenting with more complex models. Note that the forward pass could make use of e.g. a member variable or even the data itself to determine the execution path - and it can also make use of multiple arguments!

Now let's initialize the network and the optimizer.

```python id=611e988f-7f48-460d-a1f2-ff22c55955ae
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
```

Note: If we were using a GPU for training, we should have also sent the network parameters to the GPU using e.g. `network.cuda()`. It is important to transfer the network's parameters to the appropriate device before passing them to the optimizer, otherwise the optimizer will not be able to keep track of them in the right way.

# Training the Model

Time to build our training loop. First we want to make sure our network is in training mode. Then we iterate over all training data once per epoch. Loading the individual batches is handled by the DataLoader. First we need to manually set the gradients to zero using `optimizer.zero_grad()` since PyTorch by default accumulates gradients. We then produce the output of our network (forward pass) and compute a negative log-likelihodd loss between the output and the ground truth label. The `backward()` call we now collect a new set of gradients which we propagate back into each of the network's parameters using `optimizer.step()`. For more detailed information about the inner workings of PyTorch's automatic gradient system, see [the official docs for autograd](https://pytorch.org/docs/stable/notes/autograd.html#) (highly recommended).

We'll also keep track of the progress with some printouts. In order to create a nice training curve later on we also create two lists for saving training and testing losses. On the x-axis we want to display the number of training examples the network has seen during training. 

```python id=171db76f-b258-4de9-b1bf-8ba3f47044cd
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
```

We'll run our test loop once before even starting the training to see what accuracy/loss we achieve just with randomly initialized network parameters. Can you guess what our accuracy might look like for this case?

```python id=fe20d594-44f6-4282-b536-051511bbb1cd
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), '/results/model.pth')
      torch.save(optimizer.state_dict(), '/results/optimizer.pth')
```

Neural network modules as well as optimizers have the ability to save and load their internal state using `.state_dict()`. With this we can continue training from previously saved state dicts if needed - we'd just need to call `.load_state_dict(state_dict)`. 

Now for our test loop. Here we sum up the test loss and keep track of correctly classified digits to compute the accuracy of the network. 

```python id=f63b5dac-0704-4d42-b9fb-5146009c7710
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
```

Using the context manager `no_grad()` we can avoid storing the computations done producing the output of our network in the computation graph.

Time to run the training! We'll manually add a `test()` call before we loop over n_epochs to evaluate our model with randomly initialized parameters.

```python id=32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09
test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
```

[optimizer.pth][nextjournal#output#32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09#optimizer.pth]

[model.pth][nextjournal#output#32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09#model.pth]

# Evaluating the Model's Performance

And that's it. With just 3 epochs of training we already managed to achieve 97% accuracy on the test set! We started out with randomly initialized parameters and as expected only got about 10% accuracy on the test set before starting the training.

Let's plot our training curve.

```python id=12a015f7-63f8-4ebc-b500-1405333d8184
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig
```

![result][nextjournal#output#12a015f7-63f8-4ebc-b500-1405333d8184#result]

Judging from the *training curve* ([reference][nextjournal#reference#31a9f56f-a52a-4df0-9f58-729c877329ff]) it looks like we could even continue training for a few more epochs!

But before that let's again look at a few examples as we did earlier and compare the model's output.

```python id=c11b1963-b1ef-41bc-8607-d68c04c2b9b1
with torch.no_grad():
  output = network(example_data)
```

```python id=bf6ab539-81f3-4fb9-ae18-0798c44f2906
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
fig
```

![result][nextjournal#output#bf6ab539-81f3-4fb9-ae18-0798c44f2906#result]

Our model's predictions seem to be on point for those examples!

# Continued Training from Checkpoints

Now let's continue training the network, or rather see how we can continue training from the state_dicts we saved during our first training run. We'll initialize a new set of network and optimizers.

```python id=6bf2bd46-d20f-4c47-bc68-4c4bec2ebe31
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)
```

Using `.load_state_dict()` we can now load the internal state of the network and optimizer when we last saved them.

```python id=4406883b-cd4a-41d2-8988-67c195ff8d38
network_state_dict = torch.load([reference][nextjournal#reference#51705b49-5010-4fc4-9178-29f6d17b9ec6])
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load([reference][nextjournal#reference#e8f49774-b522-4455-9610-27384f3a06a6])
continued_optimizer.load_state_dict(optimizer_state_dict)
```

Again running a training loop should immediately pick up the training where we left it. To check on that let's simply use the same lists as before to keep track of the loss values. Due to the way we constructed the test counter for the number of training examples seen we manually have to append to it here.

```python id=871ab2c7-0b4c-4acc-941b-31d9e3dba36d
for i in range(4,9):
  test_counter.append(i*len(train_loader.dataset))
  train(i)
  test()
```

[optimizer.pth][nextjournal#output#871ab2c7-0b4c-4acc-941b-31d9e3dba36d#optimizer.pth]

[model.pth][nextjournal#output#871ab2c7-0b4c-4acc-941b-31d9e3dba36d#model.pth]

Great! We again see a (much slower) increase in test set accuracy from epoch to epoch. Let's visualize this to further inspect the training progress.

```python id=70fdb5a2-24ec-455c-bff3-c6a744a58a9d
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig
```

![result][nextjournal#output#70fdb5a2-24ec-455c-bff3-c6a744a58a9d#result]

That still looks like a rather smooth learning curve just as if we initially would've trained for 8 epochs! Remember that we just appended values to the same lists starting from the 5th red dot onward.

From this we can conclue two things:

1\. Continuing from the checkpointed internal state worked as intended.

2\. We still don't seem to run into overfitting issues! It looks like our dropout layers did a good job regularizing the model.

In summary we built a new environment with PyTorch and TorchVision, used it to classifiy handwritten digits from the MNIST dataset and hopefully developed a good intuition using PyTorch. For further information the official [PyTorch documentation](https://pytorch.org/docs/stable/index.html) is really nicely written and the [forums](https://discuss.pytorch.org/) are also quite active!

[signup][nextjournal#signup#69889215-2d08-4d34-aa94-59dcc1157666]


[nextjournal#file#cc8b4b29-c59f-4cae-a9bf-51888d7388f8]:
<https://nextjournal.com/data/QmcZCUUTqhqGFR4XCx6UjiPfuYjH2aE7an7We2NnvQ9bm9?content-type=image/jpeg&node-id=cc8b4b29-c59f-4cae-a9bf-51888d7388f8&filename=55154820_a43d3eb317_o.jpg&node-kind=file> (<p>cheapeeats, <a href="https://www.flickr.com/photos/cheapeats/55154820/in/photolist-5SFBb-dJHpHx-bpXtXc-bpPx1g-9nfqiq-a1p73h-EZSGK-gHE3bY-nkU9D-eza7r-cDUAqf-6cgrN3-8Y36fN-9H1M8e-8XZ3n6-6TWva-4vDaVf-6E6LFF-2Gf4z-9pikoR-bpPx1D-a1m9eB-4JAH3z-8XZ3gD-76oZKx-a1meki-6pt9WN-47Uyn-79nqaN-6pt8M9-zwmab-8vusH4-RdAKH-5viFJ9-6poZi6-6pte5J-21id5gS-wx5ep-DLpVMj-uhWoxt-LtCN2g-ahJJ49-DjTbfN-66a28U-8EUt2s-jEFPQJ-buWv9D-8tJtdS-bh6zXt-ahJKdC">Notes</a>, 2005, photograph</p>)

[nextjournal#reference#3e383ae1-aa7c-4289-9256-3a41ca01f830]:
<#nextjournal#reference#3e383ae1-aa7c-4289-9256-3a41ca01f830>

[nextjournal#output#07b8dcbb-55f8-456b-9fd0-5d49af6701dc#result]:
<https://nextjournal.com/data/QmQg8otVEBseL7CRUMZzdZ1HhouSgKijdjnpi6QtRyuDoA?content-type=image/svg%2Bxml&node-id=07b8dcbb-55f8-456b-9fd0-5d49af6701dc&node-kind=output>

[nextjournal#signup#81c8c1ec-526b-46c8-a643-9b13dbb4d111]:
<https://nextjournal.com>

[nextjournal#output#32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09#optimizer.pth]:
<https://nextjournal.com/data/QmaE3fHpAvztf8aoqxx73WmSGBCfBx8UsVkH4HR3ew2Uqf?content-type=application/octet-stream&node-id=32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09&filename=optimizer.pth&node-kind=output>

[nextjournal#output#32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09#model.pth]:
<https://nextjournal.com/data/QmNrmBowtQCxoxtD32MF2iDe1gHJWrFDqNaCixzNLtgxGp?content-type=application/octet-stream&node-id=32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09&filename=model.pth&node-kind=output>

[nextjournal#output#12a015f7-63f8-4ebc-b500-1405333d8184#result]:
<https://nextjournal.com/data/QmXjgT2GjQC34NoQWZ5CFvhAr84gawY69xdx17eJyh2smA?content-type=image/svg%2Bxml&node-id=12a015f7-63f8-4ebc-b500-1405333d8184&node-kind=output>

[nextjournal#reference#31a9f56f-a52a-4df0-9f58-729c877329ff]:
<#nextjournal#reference#31a9f56f-a52a-4df0-9f58-729c877329ff>

[nextjournal#output#bf6ab539-81f3-4fb9-ae18-0798c44f2906#result]:
<https://nextjournal.com/data/QmYwGxVMhASPZVBTdwkroHghq2QoyubdNpb1VKxcU7mvB3?content-type=image/svg%2Bxml&node-id=bf6ab539-81f3-4fb9-ae18-0798c44f2906&node-kind=output>

[nextjournal#reference#51705b49-5010-4fc4-9178-29f6d17b9ec6]:
<#nextjournal#reference#51705b49-5010-4fc4-9178-29f6d17b9ec6>

[nextjournal#reference#e8f49774-b522-4455-9610-27384f3a06a6]:
<#nextjournal#reference#e8f49774-b522-4455-9610-27384f3a06a6>

[nextjournal#output#871ab2c7-0b4c-4acc-941b-31d9e3dba36d#optimizer.pth]:
<https://nextjournal.com/data/QmRvK5Ct6KrMVa5KQcv7Nv8uEXsjf5WWwqoBqWLYgw1kAc?content-type=application/octet-stream&node-id=871ab2c7-0b4c-4acc-941b-31d9e3dba36d&filename=optimizer.pth&node-kind=output>

[nextjournal#output#871ab2c7-0b4c-4acc-941b-31d9e3dba36d#model.pth]:
<https://nextjournal.com/data/QmUsh19CBeq921XR1RPEoyFngyuAU9kCxAmfZmZTrGZ1nN?content-type=application/octet-stream&node-id=871ab2c7-0b4c-4acc-941b-31d9e3dba36d&filename=model.pth&node-kind=output>

[nextjournal#output#70fdb5a2-24ec-455c-bff3-c6a744a58a9d#result]:
<https://nextjournal.com/data/QmQJzPzsbWqGfRwcsZy4vLvDiZ5L182at23e8yAtdNQur7?content-type=image/svg%2Bxml&node-id=70fdb5a2-24ec-455c-bff3-c6a744a58a9d&node-kind=output>

[nextjournal#signup#69889215-2d08-4d34-aa94-59dcc1157666]:
<https://nextjournal.com>

<details id="com.nextjournal.article">
<summary>This notebook was exported from <a href="https://nextjournal.com/a/C6Xspits7tPKmPEwzsjm1d?change-id=CeKVumNuBHmLPUuTCTztdB">https://nextjournal.com/a/C6Xspits7tPKmPEwzsjm1d?change-id=CeKVumNuBHmLPUuTCTztdB</a></summary>

```edn nextjournal-metadata
{:article
 {:settings
  {:use-gpu? true, :image "nextjournal/ubuntu:17.04-658650854"},
  :nodes
  {"07b8dcbb-55f8-456b-9fd0-5d49af6701dc"
   {:compute-ref #uuid "f2b6dad8-821a-466e-9bec-70faac87d32d",
    :exec-duration 1179,
    :id "07b8dcbb-55f8-456b-9fd0-5d49af6701dc",
    :kind "code",
    :output-log-lines {:stdout 5},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"],
    :stdout-collapsed? true},
   "12a015f7-63f8-4ebc-b500-1405333d8184"
   {:compute-ref #uuid "7ba13a17-852b-41ed-8be4-0bc48565a457",
    :exec-duration 790,
    :id "12a015f7-63f8-4ebc-b500-1405333d8184",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "171db76f-b258-4de9-b1bf-8ba3f47044cd"
   {:compute-ref #uuid "a79c7b31-7113-42f3-b311-2bc78a2414c2",
    :exec-duration 153,
    :id "171db76f-b258-4de9-b1bf-8ba3f47044cd",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "1cda50d5-732e-4852-bec7-3cb50a07616a"
   {:compute-ref #uuid "9157aed1-04e9-4fa6-88f2-d3d0cae26386",
    :exec-duration 3271,
    :id "1cda50d5-732e-4852-bec7-3cb50a07616a",
    :kind "code",
    :name "Creating DataLoader",
    :output-log-lines {:stdout 7},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "20c1cdd6-35dd-48aa-bac9-2f31a3f37366"
   {:compute-ref #uuid "d35b695f-ef2c-421b-a388-6ca7dcbabda3",
    :exec-duration 172,
    :id "20c1cdd6-35dd-48aa-bac9-2f31a3f37366",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "23b72132-7711-4c44-ac13-8f8ac9ea284d"
   {:environment
    [:environment
     {:article/nextjournal.id
      #uuid "5b5370f0-f3da-475b-8539-83b4ea304abd",
      :change/nextjournal.id
      #uuid "5b6d5a1d-ab79-4e53-b6a7-da586cad6759",
      :node/id "eeec20d3-f1eb-4e2d-aa54-b8a9ea5bf935"}],
    :id "23b72132-7711-4c44-ac13-8f8ac9ea284d",
    :kind "runtime",
    :language "python",
    :name "PyTorch & TorchVision",
    :type :nextjournal},
   "2b7f273e-ceac-4907-806f-8279409c43ba"
   {:compute-ref #uuid "8deceb63-36ea-4cbd-811f-5919f489339c",
    :exec-duration 252,
    :id "2b7f273e-ceac-4907-806f-8279409c43ba",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "31a9f56f-a52a-4df0-9f58-729c877329ff"
   {:id "31a9f56f-a52a-4df0-9f58-729c877329ff",
    :kind "reference",
    :ref-id "12a015f7-63f8-4ebc-b500-1405333d8184"},
   "32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09"
   {:compute-ref #uuid "1d41301a-f242-4f2a-bcc2-0ea1334c385c",
    :exec-duration 105159,
    :id "32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09",
    :kind "code",
    :name "Training Loop",
    :output-log-lines {:stdout 303},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "3e383ae1-aa7c-4289-9256-3a41ca01f830"
   {:id "3e383ae1-aa7c-4289-9256-3a41ca01f830",
    :kind "reference",
    :ref-id "639bcf38-f56d-446d-a319-46d5977c1e9f"},
   "4406883b-cd4a-41d2-8988-67c195ff8d38"
   {:compute-ref #uuid "a0215745-15f5-4e3d-93f7-5db7c68e77e2",
    :exec-duration 300,
    :id "4406883b-cd4a-41d2-8988-67c195ff8d38",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "46669dc1-14ff-44bb-a645-dd92074e3b41"
   {:compute-ref #uuid "e1a1c6ec-b65f-4f99-b866-c4d603abd4d5",
    :exec-duration 499,
    :id "46669dc1-14ff-44bb-a645-dd92074e3b41",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "51705b49-5010-4fc4-9178-29f6d17b9ec6"
   {:id "51705b49-5010-4fc4-9178-29f6d17b9ec6",
    :kind "reference",
    :link
    [:output "32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09" "model.pth"]},
   "611e988f-7f48-460d-a1f2-ff22c55955ae"
   {:compute-ref #uuid "be97c00e-929d-40bc-8cd3-8031ae13abbd",
    :exec-duration 157,
    :id "611e988f-7f48-460d-a1f2-ff22c55955ae",
    :kind "code",
    :name "Network & Optimizer Setup",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "639bcf38-f56d-446d-a319-46d5977c1e9f"
   {:compute-ref #uuid "cec5db12-a622-4fb5-a855-0cc1cc2fded6",
    :exec-duration 211,
    :id "639bcf38-f56d-446d-a319-46d5977c1e9f",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "6bf2bd46-d20f-4c47-bc68-4c4bec2ebe31"
   {:compute-ref #uuid "e6eaeb22-749e-43ea-b902-55c430fceffa",
    :exec-duration 196,
    :id "6bf2bd46-d20f-4c47-bc68-4c4bec2ebe31",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "70fdb5a2-24ec-455c-bff3-c6a744a58a9d"
   {:compute-ref #uuid "a5a86985-dd09-4492-98b2-d2624ae5699a",
    :exec-duration 777,
    :id "70fdb5a2-24ec-455c-bff3-c6a744a58a9d",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "7bc8789f-80f8-4f87-bb55-5a6ece76684d"
   {:compute-ref #uuid "c7630764-e83f-4842-acc4-eb0d058908d5",
    :exec-duration 1569,
    :id "7bc8789f-80f8-4f87-bb55-5a6ece76684d",
    :kind "code",
    :name "Imports",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "871ab2c7-0b4c-4acc-941b-31d9e3dba36d"
   {:compute-ref #uuid "1e8e0a5d-3f8f-46ee-b0b0-ebbafb93cbed",
    :exec-duration 163904,
    :id "871ab2c7-0b4c-4acc-941b-31d9e3dba36d",
    :kind "code",
    :output-log-lines {:stdout 486},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "aa899593-62cf-4671-a3f5-8caa180f7344"
   {:compute-ref #uuid "6de5cefb-c7b0-472f-a03b-ac11bb470cfa",
    :exec-duration 167,
    :id "aa899593-62cf-4671-a3f5-8caa180f7344",
    :kind "code",
    :name "Creating the Network",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "bf6ab539-81f3-4fb9-ae18-0798c44f2906"
   {:compute-ref #uuid "6531999a-5b71-44cf-accf-e01610da90f7",
    :exec-duration 1035,
    :id "bf6ab539-81f3-4fb9-ae18-0798c44f2906",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"],
    :stdout-collapsed? true},
   "c11b1963-b1ef-41bc-8607-d68c04c2b9b1"
   {:compute-ref #uuid "332f5e1f-d10e-42d0-900d-93dd373f1cc6",
    :exec-duration 374,
    :id "c11b1963-b1ef-41bc-8607-d68c04c2b9b1",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "cc8b4b29-c59f-4cae-a9bf-51888d7388f8"
   {:id "cc8b4b29-c59f-4cae-a9bf-51888d7388f8", :kind "file"},
   "e8f49774-b522-4455-9610-27384f3a06a6"
   {:id "e8f49774-b522-4455-9610-27384f3a06a6",
    :kind "reference",
    :link
    [:output "32ae962e-7ed7-4fe8-9a5b-3d90b89d6b09" "optimizer.pth"]},
   "f63b5dac-0704-4d42-b9fb-5146009c7710"
   {:compute-ref #uuid "19797c25-0c49-4ddf-a875-1a9c247b0962",
    :exec-duration 167,
    :id "f63b5dac-0704-4d42-b9fb-5146009c7710",
    :kind "code",
    :name "Test Loop",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]},
   "fe20d594-44f6-4282-b536-051511bbb1cd"
   {:compute-ref #uuid "8bc28008-dc1b-4400-8d75-0954ab7365e7",
    :exec-duration 166,
    :id "fe20d594-44f6-4282-b536-051511bbb1cd",
    :kind "code",
    :name "Train Loop",
    :output-log-lines {},
    :runtime [:runtime "23b72132-7711-4c44-ac13-8f8ac9ea284d"]}},
  :nextjournal/id #uuid "59da45b4-0624-496d-b80d-f21bc7db7aa4",
  :article/change
  {:nextjournal/id #uuid "5e4a71df-c934-4cbb-a372-5c8813bc0246"}}}

```
</details>
