let myChart = null

sendData = async () => {

    $("#sendDataButton").html('Loading...')
    $("#sendDataButton").prop('disabled', true)

    let file = $("#fileUploader").prop('files')[0]
    const formData = new FormData()
    formData.append('file', file)  
    let r = await fetch('/compute', {method: "POST", body: formData})
    let data = await r.json()

    generateChart(data.result)
    console.log(data.result)
    $("#sendDataButton").prop('disabled', false)
    $("#sendDataButton").html('Send Data')

    $("#numResults").html(`${data.count} Measurements Taken`)
}

generateChart = (data) => {
    let ctx = document.getElementById('myChart').getContext('2d')
    if (myChart == null) {
        myChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    label: '# of Votes',
                    data: Object.values(data),
                    backgroundColor: Object.values(data).map( _ => "#FFFFFF")
                }]
            },  
            options: {
                scales: {
                    xAxes: [{
                        gridLines: {
                            display: false,
                        },
                        ticks: {
                            fontColor: "#FFF"
                        }
                    }],
                    yAxes: [{
                        ticks: {
                            beginAtZero: true,
                            suggestedMin: 0,
                            suggestedMax: data.count,
                            fontColor: "#FFF"
                        },
                        gridLines: {
                            display: false,
                        }
                    }]
                },
                legend: {
                    display: false
                }
            }
        })    
    }
    else {
        myChart.data.labels = Object.keys(data)
        myChart.data.datasets[0].data = Object.values(data)
        myChart.data.datasets[0].backgroundColor = Object.values(data).map( _ => "#FFFFFF")
        myChart.update()
    }
}