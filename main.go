// Copyright 2025 The SA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"image/color"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf64"
	"github.com/pointlander/sa/kmeans"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures  []float64
	Embedding []float64
	Label     string
	Cluster   int
	Index     int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
	"gen":             3,
}

// Inverse is the labels inverse map
var Inverse = [4]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
	"gen",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

// LearnEmbedding learns the embedding
func LearnEmbedding(iris []Fisher, average []float64, width int) []Fisher {
	rng := rand.New(rand.NewSource(1))
	others := tf64.NewSet()
	length := len(iris)
	if *FlagGen {
		length++
	}
	cp := make([]Fisher, length)
	copy(cp, iris)
	if *FlagGen {
		cp[len(iris)].Label = "gen"
		cp[len(iris)].Index = len(iris)
	}
	others.Add("x", 4, len(cp))
	x := others.ByName["x"]
	for _, row := range iris {
		x.X = append(x.X, row.Measures...)
	}
	if *FlagGen {
		w := x
		for i := range 4 {
			w.X = append(w.X, rng.Float64()*average[i])
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	set := tf64.NewSet()
	set.Add("i", width, len(cp))

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor*.01)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	sa := tf64.T(tf64.Mul(tf64.Dropout(tf64.MulS(set.Get("i"), set.Get("i")), dropout), tf64.T(others.Get("x"))))
	loss := tf64.Avg(tf64.Quadratic(others.Get("x"), sa))

	for iteration := range 2 * 1024 {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		set.Zero()
		others.Zero()
		l := tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return nil
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		if *FlagGen {
			w := x
			offset := len(iris) * 4
			D := w.D[offset:]
			for ii, d := range D {
				g := d * scaling
				m := B1*w.States[StateM][offset+ii] + (1-B1)*g
				v := B2*w.States[StateV][offset+ii] + (1-B2)*g*g
				w.States[StateM][offset+ii] = m
				w.States[StateV][offset+ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[offset+ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		fmt.Println(l)
	}

	if *FlagGen {
		for iteration := range 2 * 1024 {
			pow := func(x float64) float64 {
				y := math.Pow(x, float64(iteration+1))
				if math.IsNaN(y) || math.IsInf(y, 0) {
					return 0
				}
				return y
			}

			set.Zero()
			others.Zero()
			l := tf64.Gradient(loss).X[0]
			if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
				fmt.Println(iteration, l)
				return nil
			}

			norm := 0.0
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = math.Sqrt(norm)
			b1, b2 := pow(B1), pow(B2)
			scaling := 1.0
			if norm > 1 {
				scaling = 1 / norm
			}
			{
				w := x
				offset := len(iris) * 4
				D := w.D[offset:]
				for ii, d := range D {
					g := d * scaling
					m := B1*w.States[StateM][offset+ii] + (1-B1)*g
					v := B2*w.States[StateV][offset+ii] + (1-B2)*g*g
					w.States[StateM][offset+ii] = m
					w.States[StateV][offset+ii] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[offset+ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
			fmt.Println(l)
		}
	}

	meta := make([][]float64, len(cp))
	for i := range meta {
		meta[i] = make([]float64, len(cp))
	}
	const k = 3

	{
		y := set.ByName["i"]
		vectors := make([][]float64, len(cp))
		for i := range vectors {
			row := make([]float64, width)
			for ii := range row {
				row[ii] = y.X[i*width+ii]
			}
			vectors[i] = row
		}
		for i := 0; i < 33; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(meta); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range clusters {
		cp[i].Cluster = clusters[i]
	}
	for _, value := range x.X[len(iris)*4:] {
		cp[len(iris)].Measures = append(cp[len(iris)].Measures, value)
	}
	I := set.ByName["i"]
	for i := range cp {
		cp[i].Embedding = I.X[i*width : (i+1)*width]
	}
	sort.Slice(cp, func(i, j int) bool {
		return cp[i].Cluster < cp[j].Cluster
	})
	return cp
}

// Stats are the statistics for a model
type Stats struct {
	Min float64
	Max float64
}

// Network is a feedforward neural network
type Network struct {
	Rng    *rand.Rand
	Width  int
	Stats  []Stats
	Set    tf64.Set
	Others tf64.Set
	L0     tf64.Meta
	L1     tf64.Meta
	Loss   tf64.Meta
}

// NewNetwork creates a network
func NewNetwork(width int) Network {
	rng := rand.New(rand.NewSource(1))
	stats := make([]Stats, width)
	for i := range stats {
		stats[i].Min = math.MaxFloat64
		stats[i].Max = -math.MaxFloat64
	}
	set := tf64.NewSet()
	set.Add("w0", width, width)
	set.Add("b0", width)
	set.Add("w1", 2*width, 4)
	set.Add("b1", 4, 1)
	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	others := tf64.NewSet()
	others.Add("input", width)
	others.Add("output", 4)
	input := others.ByName["input"].X
	others.ByName["input"].X = input[:cap(input)]
	output := others.ByName["output"].X
	others.ByName["output"].X = output[:cap(output)]

	l0 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
	l1 := tf64.Add(tf64.Mul(set.Get("w1"), l0), set.Get("b1"))
	loss := tf64.Quadratic(others.Get("output"), l1)

	return Network{
		Rng:    rng,
		Width:  width,
		Stats:  stats,
		Set:    set,
		Others: others,
		L0:     l0,
		L1:     l1,
		Loss:   loss,
	}
}

// Learn learns with a network
func (n *Network) Learn(data []Fisher) {
	for i := range data {
		for ii := range n.Stats {
			value := data[i].Embedding[ii]
			if value > n.Stats[ii].Max {
				n.Stats[ii].Max = value
			}
			if value < n.Stats[ii].Min {
				n.Stats[ii].Min = value
			}
		}
	}
	for iteration := range 2 * 1024 {
		index := n.Rng.Intn(len(data))
		copy(n.Others.ByName["input"].X, data[index].Embedding)
		copy(n.Others.ByName["output"].X, data[index].Measures)
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		n.Set.Zero()
		n.Others.Zero()
		l := tf64.Gradient(n.Loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
		}

		norm := 0.0
		for _, p := range n.Set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range n.Set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		fmt.Println(l)
	}
}

var (
	// FlagGen generation mode
	FlagGen = flag.Bool("gen", false, "generation mode")
)

func main() {
	flag.Parse()

	iris := Load()
	average := make([]float64, 4)
	for _, row := range iris {
		for i, value := range row.Measures {
			average[i] += value
		}
	}
	for i, value := range average {
		average[i] = value / float64(len(iris))
	}

	cp := LearnEmbedding(iris, average, 2)
	acc := make(map[string][4]int)
	for i := range cp {
		fmt.Println(cp[i].Cluster, cp[i].Label)
		counts := acc[cp[i].Label]
		counts[cp[i].Cluster]++
		acc[cp[i].Label] = counts
	}

	cp5 := LearnEmbedding(iris, average, 5)
	acc5 := make(map[string][4]int)
	for i := range cp5 {
		fmt.Println(cp5[i].Cluster, cp5[i].Label)
		counts := acc5[cp5[i].Label]
		counts[cp5[i].Cluster]++
		acc5[cp5[i].Label] = counts
	}

	indexes := make([]*Fisher, len(cp))
	for i := range cp {
		indexes[cp[i].Index] = &cp[i]
	}

	points := make([]plotter.XYs, 4)
	networks, sets := make([]Network, 3), make(map[int][]Fisher, 3)
	for i := range networks {
		networks[i] = NewNetwork(5)
	}
	for i := range len(cp5) {
		embedding := indexes[cp5[i].Index].Embedding
		points[cp5[i].Cluster] = append(points[cp5[i].Cluster], plotter.XY{X: embedding[0], Y: embedding[1]})
		set := sets[cp5[i].Cluster]
		set = append(set, cp5[i])
		sets[cp5[i].Cluster] = set
	}
	p := plot.New()

	p.Title.Text = "x vs y"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	colors := make([]color.RGBA, len(points))
	colors[0] = color.RGBA{R: 255, A: 255}
	colors[1] = color.RGBA{G: 255, A: 255}
	colors[2] = color.RGBA{B: 255, A: 255}
	colors[3] = color.RGBA{R: 255, G: 255, A: 255}

	for i := range points {
		scatter, err := plotter.NewScatter(points[i])
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Color = colors[i]
		p.Add(scatter)
	}

	err := p.Save(8*vg.Inch, 8*vg.Inch, "clusters.png")
	if err != nil {
		panic(err)
	}

	for i := range networks {
		fmt.Println("--------------------------------------------------")
		networks[i].Learn(sets[i])
	}

	for i, v := range acc {
		fmt.Println(i, v)
	}
	fmt.Println()
	for i, v := range acc5 {
		fmt.Println(i, v)
	}

	fmt.Println(average)
	if *FlagGen {
		for _, value := range cp {
			if value.Label == "gen" {
				fmt.Println(value.Measures)
			}
		}
		for _, value := range cp5 {
			if value.Label == "gen" {
				fmt.Println(value.Measures)
			}
		}
	}

	inputs := make([]float64, 5)
	for i := range inputs {
		inputs[i] = (networks[0].Stats[i].Max-networks[0].Stats[i].Min)*networks[0].Rng.Float64() + networks[0].Stats[i].Min
	}
	copy(networks[0].Others.ByName["input"].X, inputs)
	networks[0].L1(func(a *tf64.V) bool {
		fmt.Println(a.X)
		return true
	})
}
