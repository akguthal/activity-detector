sendData = async () => {

    $("#sendDataButton").prop('disabled', true)
    $("#activity").html("Loading...")

    let file = $("#fileUploader").prop('files')[0]
    const formData = new FormData()
    formData.append('file', file)  
    let r = await fetch('/compute', {method: "POST", body: formData})
    let data = await r.json()

    $("#activity").html(data.result)
    $("#sendDataButton").prop('disabled', false)
}