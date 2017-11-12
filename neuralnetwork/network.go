package neuralnetwork

import(
    "fmt"
)

type Network struct {
    hiddenLayers []Layer
    outputLayer Layer
    input [][]float64
    Error float64
}

func (network *Network) Init(n_features, depth, width, n_outputs int) {
    hiddenLayers := make([]Layer, depth)
    for i := 0; i < depth; i++ {
        hiddenLayers[i] = Layer{}

        if i == 0 {
            hiddenLayers[i].Init(width, n_features)
        } else {
            hiddenLayers[i].Init(width, width)
        }
    }
    outputLayer := Layer{}
    outputLayer.Init(n_outputs, width)

    network.hiddenLayers = hiddenLayers
    network.outputLayer = outputLayer
}

func (network *Network) Data(input [][]float64) {
    network.input = input
}

// Calculates Cost Function - expecting 1 output
func (network *Network) Cost(errorFunc errorFunction, actual [][]float64) {
    network.Error = errorFunc(network.outputLayer.output, actual)
}

// Calculates Cost Function - expecting 1 output
func (network *Network) Predict(input [][]float64) [][]float64 {
   network.input = input
   ForwardPropogate(network, "sigmoid")
   network.Cost(SimpleCost, input) 
   fmt.Println("Error:", network.Error)
   return network.outputLayer.Output()
}