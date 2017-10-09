package neural

type Layer struct {
    neurons []Neuron
}

func (layer *Layer) Init(num_neuron, neuron_size int) {
    layer.neurons = make([]Neuron, num_neuron)
    for i := 0; i < num_neuron; i++ {
        layer.neurons[i].Init(neuron_size)
    }
}

type Network struct {
    hiddenLayers []Layer
    outputLayer Layer
    input [][]float64
}

func (network *Network) Init(n_inputs, depth, width, n_outputs int) {
    hiddenLayers := make([]Layer, depth)
    for i := 0; i < depth; i++ {
        hiddenLayers[i] = Layer{}

        if i == 0 {
            hiddenLayers[i].Init(width, n_inputs)
        } else {
            hiddenLayers[i].Init(width, width)
        }
    }
    outputLayer := Layer{}
    outputLayer.Init(n_outputs, width)

    network.hiddenLayers = hiddenLayers
    network.outputLayer = outputLayer
}
