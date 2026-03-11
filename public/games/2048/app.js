document.addEventListener("DOMContentLoaded", () => {
    const gridDisplay = document.querySelector(".grid")
    const scoreDisplay = document.querySelector("#score")
    const resultDisplay = document.querySelector("#result")
    const width = 4
    let squares = []
    let score = 0

    // create the playing board
    function createBoard() {
        for (let i = 0; i < width * width; i++) {
            const square = document.createElement("div")
            square.innerHTML = 0
            square.setAttribute("data-value", 0)
            gridDisplay.appendChild(square)
            squares.push(square)
        }
        generate()
        generate()
    }
    createBoard()

    //generate a new number
    function generate() {
        const randomNumber = Math.floor(Math.random() * squares.length)
        if (squares[randomNumber].innerHTML == 0) {
            squares[randomNumber].innerHTML = 2
            squares[randomNumber].setAttribute("data-value", 2)
            // Add animation class
            squares[randomNumber].classList.add("pop")
            setTimeout(() => {
                squares[randomNumber].classList.remove("pop")
            }, 200)
            checkForGameOver()
        } else generate()
    }

    function moveRight() {
        for (let i = 0; i < 16; i++) {
            if (i % 4 === 0) {
                let totalOne = squares[i].innerHTML
                let totalTwo = squares[i + 1].innerHTML
                let totalThree = squares[i + 2].innerHTML
                let totalFour = squares[i + 3].innerHTML
                let row = [parseInt(totalOne), parseInt(totalTwo), parseInt(totalThree), parseInt(totalFour)]

                let filteredRow = row.filter(num => num)
                let missing = 4 - filteredRow.length
                let zeros = Array(missing).fill(0)
                let newRow = zeros.concat(filteredRow)

                updateSquare(i, newRow[0])
                updateSquare(i + 1, newRow[1])
                updateSquare(i + 2, newRow[2])
                updateSquare(i + 3, newRow[3])
            }
        }
    }

    function moveLeft() {
        for (let i = 0; i < 16; i++) {
            if (i % 4 === 0) {
                let totalOne = squares[i].innerHTML
                let totalTwo = squares[i + 1].innerHTML
                let totalThree = squares[i + 2].innerHTML
                let totalFour = squares[i + 3].innerHTML
                let row = [parseInt(totalOne), parseInt(totalTwo), parseInt(totalThree), parseInt(totalFour)]

                let filteredRow = row.filter(num => num)
                let missing = 4 - filteredRow.length
                let zeros = Array(missing).fill(0)
                let newRow = filteredRow.concat(zeros)

                updateSquare(i, newRow[0])
                updateSquare(i + 1, newRow[1])
                updateSquare(i + 2, newRow[2])
                updateSquare(i + 3, newRow[3])
            }
        }
    }

    function moveUp() {
        for (let i = 0; i < 4; i++) {
            let totalOne = squares[i].innerHTML
            let totalTwo = squares[i + width].innerHTML
            let totalThree = squares[i + width * 2].innerHTML
            let totalFour = squares[i + width * 3].innerHTML
            let column = [parseInt(totalOne), parseInt(totalTwo), parseInt(totalThree), parseInt(totalFour)]

            let filteredColumn = column.filter(num => num)
            let missing = 4 - filteredColumn.length
            let zeros = Array(missing).fill(0)
            let newColumn = filteredColumn.concat(zeros)

            updateSquare(i, newColumn[0])
            updateSquare(i + width, newColumn[1])
            updateSquare(i + width * 2, newColumn[2])
            updateSquare(i + width * 3, newColumn[3])
        }
    }

    function moveDown() {
        for (let i = 0; i < 4; i++) {
            let totalOne = squares[i].innerHTML
            let totalTwo = squares[i + width].innerHTML
            let totalThree = squares[i + width * 2].innerHTML
            let totalFour = squares[i + width * 3].innerHTML
            let column = [parseInt(totalOne), parseInt(totalTwo), parseInt(totalThree), parseInt(totalFour)]

            let filteredColumn = column.filter(num => num)
            let missing = 4 - filteredColumn.length
            let zeros = Array(missing).fill(0)
            let newColumn = zeros.concat(filteredColumn)

            updateSquare(i, newColumn[0])
            updateSquare(i + width, newColumn[1])
            updateSquare(i + width * 2, newColumn[2])
            updateSquare(i + width * 3, newColumn[3])
        }
    }

    // Update square with animation
    function updateSquare(index, value) {
        const oldValue = parseInt(squares[index].innerHTML)
        squares[index].innerHTML = value
        squares[index].setAttribute("data-value", value)
        
        // Add merged animation if value increased
        if (value > oldValue && oldValue !== 0) {
            squares[index].classList.add("merged")
            setTimeout(() => {
                squares[index].classList.remove("merged")
            }, 200)
        }
    }

    function combineRow() {
        for (let i = 0; i < 15; i++) {
            if (squares[i].innerHTML === squares[i + 1].innerHTML) {
                let combinedTotal = parseInt(squares[i].innerHTML) + parseInt(squares[i + 1].innerHTML)
                if (squares[i].innerHTML !== "0") {
                    updateSquare(i, combinedTotal)
                    updateSquare(i + 1, 0)
                    score += combinedTotal
                    scoreDisplay.innerHTML = score
                }
            }
        }
        checkForWin()
    }

    function combineColumn() {
        for (let i = 0; i < 12; i++) {
            if (squares[i].innerHTML === squares[i + width].innerHTML) {
                let combinedTotal = parseInt(squares[i].innerHTML) + parseInt(squares[i + width].innerHTML)
                if (squares[i].innerHTML !== "0") {
                    updateSquare(i, combinedTotal)
                    updateSquare(i + width, 0)
                    score += combinedTotal
                    scoreDisplay.innerHTML = score
                }
            }
        }
        checkForWin()
    }

    ///assign functions to keys
    function control(e) {
        if (e.key === "ArrowLeft") {
            e.preventDefault()
            keyLeft()
        } else if (e.key === "ArrowRight") {
            e.preventDefault()
            keyRight()
        } else if (e.key === "ArrowUp") {
            e.preventDefault()
            keyUp()
        } else if (e.key === "ArrowDown") {
            e.preventDefault()
            keyDown()
        }
    }
    document.addEventListener("keydown", control)

    function keyLeft() {
        moveLeft()
        combineRow()
        moveLeft()
        generate()
    }

    function keyRight() {
        moveRight()
        combineRow()
        moveRight()
        generate()
    }

    function keyUp() {
        moveUp()
        combineColumn()
        moveUp()
        generate()
    }

    function keyDown() {
        moveDown()
        combineColumn()
        moveDown()
        generate()
    }

    //check for the number 2048 in the squares to win
    function checkForWin() {
        for (let i = 0; i < squares.length; i++) {
            if (squares[i].innerHTML == 2048) {
                resultDisplay.innerHTML = "🎉 You WIN!"
                document.removeEventListener("keydown", control)
                setTimeout(clear, 3000)
            }
        }
    }

    //check if there are no zeros on the board to lose
    function checkForGameOver() {
        let zeros = 0
        for (let i = 0; i < squares.length; i++) {
            if (squares[i].innerHTML == 0) {
                zeros++
            }
        }
        if (zeros === 0) {
            resultDisplay.innerHTML = "😢 You LOSE!"
            document.removeEventListener("keydown", control)
            setTimeout(clear, 3000)
        }
    }

    function clear() {
        clearInterval(myTimer)
    }

    //add colours
    function addColours() {
        for (let i = 0; i < squares.length; i++) {
            const value = parseInt(squares[i].innerHTML)
            squares[i].setAttribute("data-value", value)
        }
    }
    addColours()

    let myTimer = setInterval(addColours, 50)
})
