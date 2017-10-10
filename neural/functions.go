package neural

import(
    "math"
    "fmt"
)

// Activation functions
func Sigmoid(input float64) float64 {
    return 1.0/(1.0+math.Exp(input))
}

func Relu(input float64) float64 {
    return math.Max(0, input)
}

func Activate(input [][]float64, weights []float64) ([]float64, bool) {
    fmt.Println("TEST", input, weights, "\n")
    if input == nil || weights == nil || (len(input[0])+1 != len(weights)) {
        return nil, false
    }

    n_dim := len(input)
    m_dim := len(input[0])
    res := make([]float64, n_dim)
    for i := 0; i < n_dim; i++ {
        val := 0.0
        for j := 0; j < m_dim; j++ {
            val += input[i][j]*weights[j]
        }
        // Add bias weight to result
        res[i] = val+weights[m_dim]
    }
        fmt.Println("TEST", res, "\n")
    return res, true
}
