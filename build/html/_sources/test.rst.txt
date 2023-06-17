.. code:: ipython3

    import torch

.. code:: ipython3

    print(torch.cuda.is_available())


.. parsed-literal::

    True


.. code:: ipython3

    x = torch.tensor([1,2,3], device="cuda:0")
    print(x.device)


.. parsed-literal::

    cuda:0


.. code:: ipython3

    x




.. parsed-literal::

    tensor([1, 2, 3], device='cuda:0')



.. code:: ipython3

    print("Using torch", torch.__version__)


.. parsed-literal::

    Using torch 2.0.0


.. code:: ipython3

    print(torch.cuda.device_count())


.. parsed-literal::

    1


.. code:: ipython3

    print(torch.cuda.get_device_name(0))


.. parsed-literal::

    NVIDIA GeForce RTX 4090


Real Test

.. code:: ipython3

    import numpy as np
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype = np.float32)
    x_train = x_train.reshape(-1,1)
    x_train.shape




.. parsed-literal::

    (11, 1)



.. code:: ipython3

    y_values = [2*i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)
    y_train.shape




.. parsed-literal::

    (11, 1)



.. code:: ipython3

    import torch.nn as nn

.. code:: ipython3

    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
    
        def forward(self, x):
            out = self.linear(x)
            return out

.. code:: ipython3

    input_dim = 1
    output_dim = 1
    model = LinearRegressionModel(input_dim, output_dim)

Send to cuda

.. code:: ipython3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)




.. parsed-literal::

    LinearRegressionModel(
      (linear): Linear(in_features=1, out_features=1, bias=True)
    )



.. code:: ipython3

    epochs = 1000
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

.. code:: ipython3

    for epoch in range(epochs):
        epoch += 1
    
        # .to(device) send to cuda
        inputs = torch.from_numpy(x_train).to(device)
        labels = torch.from_numpy(y_train).to(device)
    
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))


.. parsed-literal::

    epoch 50, loss 0.40827712416648865
    epoch 100, loss 0.2328658550977707
    epoch 150, loss 0.13281798362731934
    epoch 200, loss 0.0757543221116066
    epoch 250, loss 0.04320744052529335
    epoch 300, loss 0.024643903598189354
    epoch 350, loss 0.014055978506803513
    epoch 400, loss 0.008016953244805336
    epoch 450, loss 0.004572576377540827
    epoch 500, loss 0.0026080235838890076
    epoch 550, loss 0.001487524015828967
    epoch 600, loss 0.0008484188001602888
    epoch 650, loss 0.00048391061136499047
    epoch 700, loss 0.0002760067582130432
    epoch 750, loss 0.00015742452524136752
    epoch 800, loss 8.978872938314453e-05
    epoch 850, loss 5.121196591062471e-05
    epoch 900, loss 2.920936640293803e-05
    epoch 950, loss 1.6659612811054103e-05
    epoch 1000, loss 9.501958629698493e-06


.. code:: ipython3

    # predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
    # predicted

.. code:: ipython3

    torch.save(model.state_dict(), 'model.pkl')

.. code:: ipython3

    model.load_state_dict(torch.load('model.pkl'))




.. parsed-literal::

    <All keys matched successfully>


