package neuralnetwork

func BackPropagate(network *Network, actual [][]float64, transfer string) bool {
    network.Cost(SimpleCostD, actual)

    // output layer error
    for i, neurons := 0, len(network.outputLayer.neurons); i < neurons; i++ {
        network.outputLayer.neurons[i].error = network.Error * transferFuncD(network.outputLayer.output[i][0], transfer)
    }
    
    // backpropagate rest of layers
    lastLayer := len(network.hiddenLayers)-1
    backPropagate(&network.hiddenLayers[lastLayer], &network.outputLayer, transfer)

    for i := lastLayer; i > 0; i-- {
        backPropagate(&network.hiddenLayers[i-1], &network.hiddenLayers[i], transfer)
    }

    return true
}

func UpdateWeights(network *Network, alpha float64) {
    for i, layers := 0, len(network.hiddenLayers); i < layers; i++ {
        // input is output from previous layer
        var input [][]float64
        if i == 0 {
            input = network.input
        } else {
            input = network.hiddenLayers[i-1].output
        }
        updateWeights(&network.hiddenLayers[i], alpha, input)
    }
    // update output layer
    updateWeights(&network.outputLayer, alpha, network.hiddenLayers[len(network.hiddenLayers)-1].output)
}

func updateWeights(layer *Layer, alpha float64, input [][]float64) {
    for _, neuron := range layer.neurons {
        for j, inputLength := 0, len(input); j < inputLength; j++ {
            for k, featLength := 0, len(input[j]); k < featLength; k++ {
                neuron.weights[k] += alpha * neuron.error * input[j][k]
            }
            // update bias
            neuron.weights[len(input[j])] += alpha * neuron.error
        }
    }
}

func backPropagate(layer *Layer, nextLayer *Layer, transfer string) {
    for i, neurons := 0, len(layer.neurons); i < neurons; i++ {
        var error float64 = 0
        for j, nextNeuron := 0, len(nextLayer.neurons); j < nextNeuron; j++{
            error += nextLayer.neurons[j].weights[i] * nextLayer.neurons[j].error
        }
        layer.neurons[i].error = error * transferFuncD(layer.output[i][0], transfer)
    }
}

func transferFuncD(output float64, transfer string) float64{
    var function func(input float64) float64
    switch transfer {
    case "relu":
        function = ReluD
    case "sigmoid":
        function = SigmoidD
    }
    return function(output)
}