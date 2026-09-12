package main

import (
	"fmt"
	"math/rand"
)

func main() {
	var n int
	var x int
	n = rand.Intn(100)
	for {
		var temp int
		x++
		fmt.Printf("Enter the Number ")
		fmt.Scanf("%v\n", &temp)
		if temp < n {
			fmt.Printf("Too low!\n")
		} else if temp > n {
			fmt.Printf("Too High!\n")
		} else {
			fmt.Printf("You got the Number %d in %d Attempts\n", n, x)
			break
		}
		fmt.Printf("Attempt - %v\n", x)
	}
}
