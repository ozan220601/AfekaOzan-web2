// network.js - הארכיטקטורה של הרשת העצבית

// 1. פונקציית הפעלה ReLU
class ReLU {
    static forward(input) {
        // מעבר על כל הערכים במערך - איפוס ערכים שליליים
        return input.map(val => Math.max(0, val));
    }
    
    static backward(input, gradient) {
        // נשתמש בזה מאוחר יותר בשלב ה-Backpropagation
        // הנגזרת של ReLU היא 1 עבור מספרים חיוביים, ו-0 עבור שליליים
        return input.map((val, i) => val > 0 ? gradient[i] : 0);
    }
}

// 2. שכבת קונבולוציה (Convolution Layer)
class ConvLayer {
    constructor(numFilters, filterSize) {
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        
        // הוספנו את המושגים מהמילון בצורה קשיחה
        this.stride = 1; 
        this.padding = 0; 
        
        this.filters = [];
        
        for (let i = 0; i < numFilters; i++) {
            let filter = new Array(filterSize * filterSize);
            for (let j = 0; j < filter.length; j++) {
                filter[j] = Math.random() - 0.5; 
            }
            this.filters.push(filter);
        }
    }

    forward(input, inputSize) {
        this.lastInput = input; 
        
        // הנוסחה המלאה והנכונה שכוללת Padding ו-Stride:
        // Output = (Input - Filter + 2*Padding) / Stride + 1
        const outputSize = Math.floor((inputSize - this.filterSize + 2 * this.padding) / this.stride) + 1;
        let output = [];

        // מעבר על כל פילטר שהגדרנו (למשל 4 פילטרים)
        for (let f = 0; f < this.numFilters; f++) {
            const currentFilter = this.filters[f];
            
            // יצירת מערך ריק עבור מפת התכונות (Feature Map) של הפילטר הנוכחי
            let featureMap = new Array(outputSize * outputSize).fill(0);

            // סריקת התמונה (החלקת הפילטר על גבי הקלט, בלולאות מקוננות)
            for (let y = 0; y < outputSize; y++) {
                for (let x = 0; x < outputSize; x++) {
                    let sum = 0;
                    
                    // הכפלת ערכי הפילטר בפיקסלים המתאימים בקלט
                    for (let fy = 0; fy < this.filterSize; fy++) {
                        for (let fx = 0; fx < this.filterSize; fx++) {
                            // מציאת המיקומים המדויקים במערכים החד-ממדיים שלנו
                            const inputIndex = ((y + fy) * inputSize) + (x + fx);
                            const filterIndex = fy * this.filterSize + fx;
                            sum += input[inputIndex] * currentFilter[filterIndex];
                        }
                    }
                    
                    // שמירת סכום המכפלות במפת התכונות
                    featureMap[y * outputSize + x] = sum;
                }
            }
            // מוסיפים את המפה המוכנה לרשימת התוצאות
            output.push(featureMap);
        }
        
        // מחזירים את התוצאה ואת הגודל החדש כדי שהשכבה הבאה תדע עם מה לעבוד
        return { output, outputSize };
    }
}

// 3. שכבת צמצום (Max Pooling Layer)
class MaxPoolingLayer {
    constructor(poolSize) {
        this.poolSize = poolSize; // לרוב גודל 2 (כלומר חלון 2x2)
    }

    forward(inputs, inputSize) {
        // inputs הוא מערך של מפות תכונות (Feature Maps) שקיבלנו מהקונבולוציה
        const outputSize = Math.floor(inputSize / this.poolSize);
        let pooledOutputs = [];

        // מעבר על כל מפת תכונות
        for (let f = 0; f < inputs.length; f++) {
            const currentMap = inputs[f];
            let pooledMap = new Array(outputSize * outputSize).fill(0);

            // סריקת המפה בקפיצות בגודל ה-Pooling
            for (let y = 0; y < outputSize; y++) {
                for (let x = 0; x < outputSize; x++) {
                    let maxVal = -Infinity;
                    
                    // חיפוש הערך המקסימלי בתוך החלון הנוכחי (2x2)
                    for (let py = 0; py < this.poolSize; py++) {
                        for (let px = 0; px < this.poolSize; px++) {
                            const index = ((y * this.poolSize + py) * inputSize) + (x * this.poolSize + px);
                            if (currentMap[index] > maxVal) {
                                maxVal = currentMap[index];
                            }
                        }
                    }
                    // שומרים רק את הפיקסל החזק ביותר
                    pooledMap[y * outputSize + x] = maxVal;
                }
            }
            pooledOutputs.push(pooledMap);
        }
        // מחזירים את התוצאות והגודל החדש לשכבה הבאה
        return { output: pooledOutputs, outputSize };
    }
}

// 4. שכבה מחוברת במלואה (Dense / Fully Connected Layer)
class DenseLayer {
    constructor(inputLength, numNeurons) {
        this.inputLength = inputLength;
        this.numNeurons = numNeurons; // בתרגיל שלנו זה יהיה 3 (עיגול, ריבוע, משולש)
        
        // אתחול משקלים (Weights) והטיות (Biases)
        this.weights = [];
        this.biases = new Array(numNeurons).fill(0);

        for (let i = 0; i < numNeurons; i++) {
            let neuronWeights = new Array(inputLength);
            for (let j = 0; j < inputLength; j++) {
                // מתחילים עם משקלים אקראיים קטנים
                neuronWeights[j] = Math.random() - 0.5;
            }
            this.weights.push(neuronWeights);
        }
    }

    forward(inputVector) {
        this.lastInput = inputVector; // נשמור לטובת הלמידה לאחור בהמשך
        let output = new Array(this.numNeurons).fill(0);

        // חישוב הניקוד לכל נוירון: מכפלת הקלט במשקלים + ההטיה
        for (let i = 0; i < this.numNeurons; i++) {
            let sum = this.biases[i];
            for (let j = 0; j < this.inputLength; j++) {
                sum += inputVector[j] * this.weights[i][j];
            }
            output[i] = sum;
        }
        return output; // מחזיר 3 מספרים (ניקוד לכל צורה)
    }
}

// 5. פונקציית Softmax להמרת הניקוד להסתברויות
class Softmax {
    static forward(input) {
        // טריק מתמטי: מציאת המקסימום כדי למנוע פיצוץ מתמטי בחישוב האקספוננט
        const maxVal = Math.max(...input);
        
        // חישוב e בחזקת הניקוד
        const exps = input.map(val => Math.exp(val - maxVal));
        
        // סכום כל הערכים כדי שנוכל לחלק בהם ולנרמל ל-100%
        const sumExps = exps.reduce((a, b) => a + b, 0);
        
        // החזרת מערך של הסתברויות בין 0 ל-1, שהסכום שלהן הוא 1
        return exps.map(val => val / sumExps);
    }
}

// 6. מחלקת המודל המרכזית - משודרגת עם שכבה נסתרת והפצה לאחור עמוקה
class CNNModel {
    constructor(numFilters, filterSize, hiddenNeurons, learningRate) {
        this.learningRate = learningRate;
        
        // שכבות חילוץ התכונות
        this.conv = new ConvLayer(numFilters, filterSize);
        this.pool = new MaxPoolingLayer(2); 
        
        // חישובי מידות הרשת
        const afterConvSize = 28 - filterSize + 1; 
        const afterPoolSize = Math.floor(afterConvSize / 2);
        const denseInputLength = numFilters * afterPoolSize * afterPoolSize;
        
        // יצירת רשת נוירונים עמוקה: קלט משוטח -> שכבה נסתרת -> שכבת פלט
        this.dense1 = new DenseLayer(denseInputLength, hiddenNeurons); // השכבה הנסתרת שהמשתמש קובע
        this.dense2 = new DenseLayer(hiddenNeurons, 3); // שכבת הסיווג הסופית (3 צורות)
    }

    /**
     * מטעמי יעילות וסיבוכיות חישובית בדפדפן הלקוח (ללא ספריות עזר), המודל משתמש בטכניקת "Random Convolutional Features".
     * שכבת הקונבולוציה מאותחלת עם פילטרים אקראיים ופועלת כמחלצת תכונות מרחביות (Feature Extractor) סטטית.
     * הלמידה עצמה וההפצה לאחור (Backpropagation) מתבצעות בשכבות העמוקות המחוברות במלואן (Dense Layers),
     * שם מתבצע עדכון המשקולות כדי לסווג את התכונות שחולצו.
     */
    trainStep(inputImage, targetLabel) {
        // --- 1. Forward Pass ---
        const convRes = this.conv.forward(inputImage, 28);
        const poolRes = this.pool.forward(convRes.output, convRes.outputSize);
        
        // השטחה
        let flattened = [];
        for (let i = 0; i < poolRes.output.length; i++) {
            flattened = flattened.concat(poolRes.output[i]);
        }
        
        // העברה בשכבות הנוירונים (Dense 1 -> ReLU -> Dense 2)
        const dense1Out = this.dense1.forward(flattened);
        const reluOut = ReLU.forward(dense1Out);
        const dense2Out = this.dense2.forward(reluOut);
        const probabilities = Softmax.forward(dense2Out);
        
        // חישוב שגיאה
        let loss = -Math.log(probabilities[targetLabel] + 1e-7);
        let predictedLabel = probabilities.indexOf(Math.max(...probabilities));
        let isCorrect = (predictedLabel === targetLabel);

        // --- 2. Backpropagation (הפצה לאחור - Chain Rule) ---
        // שגיאה בשכבת הפלט
        let dOut2 = [...probabilities];
        dOut2[targetLabel] -= 1; 
        
        // מערך לשמירת השגיאה שתעבור אחורה לשכבה הנסתרת
        let dDense1_out = new Array(this.dense1.numNeurons).fill(0);
        
        // עדכון משקלים לשכבה הסופית (Dense 2)
        for (let i = 0; i < this.dense2.numNeurons; i++) {
            for (let j = 0; j < this.dense2.inputLength; j++) {
                // צבירת השגיאה אחורה לשכבה הקודמת
                dDense1_out[j] += dOut2[i] * this.dense2.weights[i][j];
                
                // עדכון המשקל הנוכחי
                let gradient = dOut2[i] * this.dense2.lastInput[j]; 
                this.dense2.weights[i][j] -= this.learningRate * gradient;
            }
            this.dense2.biases[i] -= this.learningRate * dOut2[i];
        }
        
        // העברת השגיאה אחורה דרך פונקציית ההפעלה ReLU
        let dDense1_in = ReLU.backward(dense1Out, dDense1_out);
        
        // עדכון משקלים לשכבה הנסתרת (Dense 1)
        for (let i = 0; i < this.dense1.numNeurons; i++) {
            for (let j = 0; j < this.dense1.inputLength; j++) {
                let gradient = dDense1_in[i] * this.dense1.lastInput[j];
                this.dense1.weights[i][j] -= this.learningRate * gradient;
            }
            this.dense1.biases[i] -= this.learningRate * dDense1_in[i];
        }
        
        return { loss: loss, correct: isCorrect };
    }
    
    predict(inputImage) {
        const convRes = this.conv.forward(inputImage, 28);
        const poolRes = this.pool.forward(convRes.output, convRes.outputSize);
        
        let flattened = [];
        for (let i = 0; i < poolRes.output.length; i++) {
            flattened = flattened.concat(poolRes.output[i]);
        }
        
        const dense1Out = this.dense1.forward(flattened);
        const reluOut = ReLU.forward(dense1Out);
        const dense2Out = this.dense2.forward(reluOut);
        
        return Softmax.forward(dense2Out);
    }
    
    exportWeights() {
        return {
            dense1Weights: this.dense1.weights,
            dense1Biases: this.dense1.biases,
            dense2Weights: this.dense2.weights,
            dense2Biases: this.dense2.biases,
            convFilters: this.conv.filters
        };
    }

    importWeights(data) {
        if (!data.dense1Weights) return; // הגנה מפני טעינת שמירה ישנה מהגרסה הקודמת
        this.dense1.weights = data.dense1Weights;
        this.dense1.biases = data.dense1Biases;
        this.dense2.weights = data.dense2Weights;
        this.dense2.biases = data.dense2Biases;
        this.conv.filters = data.convFilters;
    }
}