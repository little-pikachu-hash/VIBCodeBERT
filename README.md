# Improving Fine-tuning Pre-trained Models on Small Source Code Datasets via Variational Information Bottleneck

This repository contains code and data for the paper "Improving Fine-tuning Pre-trained Models on Small Source Code Datasets via Variational Information Bottleneck".

Waiting for updates...

# Datasets involved in this work can be found as follows:

## Code-Comment Coherence

Determine whether there is "coherence" between a given method and is corresponding lead comment, that is,
whether the comment is describtive of the method. 
We use the [dataset](http://www2.unibas.it/gscanniello/coherence/) by Corazza et. al.

### Example

```java
/**
* Returns the current number of milk units in
* the inventory.
* @return int
Code-Comment Coherence
*/
Prediction [25]
public int getMilk() {
  return milk;
}
```
*Label:*: positive (coherent) 

```java
/**
   * Check inventory user interface that processes input.
   */
public static void checkInventory() {
  System.out.println(coffeeMaker.checkInventory());
  mainMenu();
}
```
**Label:** negative (incoherent) 

## Linguistic Smell Detection

Detect linguistic smells in code, that is misleading
identifier names or the violations of common naming conventions. 
Our work is based on the [dataset](https://github.com/Smfakhoury/SANER-2018-KeepItSimple-) by Fakhoury et. al.

### Example

```java
public void ToSource(StringBuilder sb) {
  sb.append(";");
  this.NewLine(sb);
}
``` 
**Label:** smelly (transform method does not return)

## Code complexity classification

Classify the algorithmic complexity of various algorithm implementations (e.g., O(1), O(n*log(n)) etc.).
We use the [dataset](https://github.com/midas-research/corcod-dataset) by Sikka et. al.

### Example

```java
class GFG {
  static int minJumps(int arr[], int n) {
    int[] jumps = new int[n];
    int min;
    jumps[n - 1] = 0;
    for (int i = n - 2; i >= 0; i--) {
      if (arr[i] == 0)
        jumps[i] = Integer.MAX_VALUE;
      else if (arr[i] >= n - i - 1) 
        jumps[i] = 1;
      else { ... }
    }
    return jumps[0];
  }
  public static void main(String[] args) {...}
}
``` 
**Label:** O(n log n)

## Code readability classification

Given a piece of code, classify it as either "readable" or "not readable".
Our work relies on the [dataset](https://dibt.unimol.it/report/readability/) by Scalabrino et. al.

### Example

```java
@Override
public void configure(Configuration cfg) {
  super.configure(cfg);
  cfg.setProperty(Environment.USE_SECOND_LEVEL_CACHE, ...);
  cfg.setProperty(Environment.GENERATE_STATISTICS, ...);
  cfg.setProperty(Environment.USE_QUERY_CACHE, "false" );
  ... // more cfg.setProperty calls
}
``` 
**Label:** readable
