package neural

type Layer struct {
    neurons []Neuron
    output [][]float64
}

func (layer *Layer) Init(num_neuron, neuron_size int) {
    layer.neurons = make([]Neuron, num_neuron)
    for i := 0; i < num_neuron; i++ {
        layer.neurons[i].Init(neuron_size)
    }

    // layer.output = make([][]float64, n_samples)
    // for i := 0; i < n_samples; i++ {
    //     lauyer.output = make([]float64, num_neuron)
    // }
}
