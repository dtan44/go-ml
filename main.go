package main

import(
    "fmt"
    "github.com/dtan44/go-ml/neuralnetwork"
)

func main() {
    // var testNeuron neuralnetwork.Neuron
    // testNeuron.Init(5)
    // fmt.Println("Neuron", testNeuron, "\n")

    // var testLayer neuralnetwork.Layer
    // testLayer.Init(5, 5)
    // fmt.Println("Layer", testLayer, "\n")

    var testNetwork neuralnetwork.Network
    testNetwork.Init(2, 4, 2, 1)
    fmt.Println("Network", testNetwork, "\n")

    var input [][]float64 = [][]float64{
        []float64{1,0},
        []float64{0,1},
        []float64{1,1},
        []float64{0,0}}

    testNetwork.Data(input)

    for i := 0; i < 30; i++ {
        neuralnetwork.ForwardPropogate(&testNetwork, "sigmoid")
        testNetwork.Cost(neuralnetwork.SimpleCost, [][]float64{[]float64{1},[]float64{1},[]float64{0},[]float64{0}})
        fmt.Println("Error", testNetwork.Error, "at Epoch ", i)

        neuralnetwork.BackPropagate(&testNetwork, [][]float64{[]float64{1},[]float64{1},[]float64{0},[]float64{0}}, "sigmoid")
        neuralnetwork.UpdateWeights(&testNetwork, 0.9)
    }
    
    fmt.Println("Network:", testNetwork, "\n")

    fmt.Println("Results:", testNetwork.Predict([][]float64{[]float64{1},[]float64{1},[]float64{0},[]float64{0}}))
}
