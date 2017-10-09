package main

import(
    "fmt"
    "github.com/dtan44/ml/neural"
)

func main() {
    var testNeuron neural.Neuron
    testNeuron.Init(5)
    fmt.Println("Neuron", testNeuron, "\n")

    var testLayer neural.Layer
    testLayer.Init(5, 5)
    fmt.Println("Layer", testLayer, "\n")

    var testNetwork neural.Network
    testNetwork.Init(2, 1, 2, 2)
    fmt.Println("Network", testNetwork, "\n")
}
