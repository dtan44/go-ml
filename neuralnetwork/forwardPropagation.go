package neuralnetwork

func ForwardPropogate(network *Network, transfer string) bool {
    // activate first layer
    if !forwardPropagateLayer(&network.hiddenLayers[0], network.input) {
        return false
    }

    transferFunc(&network.hiddenLayers[0], transfer)

    // activate hidden layers
    length := len(network.hiddenLayers)
    for i := 1; i < length; i++ {
        if !forwardPropagateLayer(&network.hiddenLayers[i],
            network.hiddenLayers[i-1].output) {
            return false
        }
        transferFunc(&network.hiddenLayers[i], transfer)
    }

    // activate output layer
    if !forwardPropagateLayer(&network.outputLayer,
        network.hiddenLayers[length-1].output) {
        return false
    }
    transferFunc(&network.outputLayer, transfer)

    return true
}

// need to optimize for caching and goroutines
func forwardPropagateLayer(layer *Layer, input [][]float64) bool {
    n_dim := len(input)
    m_dim := len(layer.neurons)
    layer.output = make([][]float64, n_dim)

    for i := 0; i < n_dim; i++ {
        layer.output[i] = make([]float64, m_dim)
    }
    for i := 0; i < m_dim; i++ {
        output, check := Activate(input, layer.neurons[i].weights)
        if !check {
            return false
        }
        for j := 0; j < n_dim; j++ {
            layer.output[j][i] = output[j]
        }
    }
    return true
}

func transferFunc(layer *Layer, transfer string) {
    var function func(input float64) float64
    switch transfer {
    case "relu":
        function = Relu
    case "sigmoid":
        function = Sigmoid
    }
    for i, length := 0, len(layer.output); i < length; i++ {
        for j, innerLength := 0, len(layer.output[i]); j < innerLength; j++ {
            layer.output[i][j] = function(layer.output[i][j])
        }
    }
}
