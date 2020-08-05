const imageUpload = document.getElementById('imageUpload')
const loaderDiv = document.getElementById('loader')

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models') //heavier more accurate version of tiny face detector
]).then(start)

function start() {
    document.body.append('Models Loaded')
    recognizeFaces()
}

async function recognizeFaces() {
    const container = document.createElement('div')
    container.style.position = 'relative'
    document.body.append(container)

    const labeledDescriptors = await loadLabeledImages()
    loaderDiv.innerHTML = 'Faces Loaded'
    imageUpload.disabled = false
    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6)

    let image
    let canvas

    //detect faces in image
    imageUpload.addEventListener('change', async () => {

        if(image) image.remove()
        if(canvas) canvas.remove()

        image = await faceapi.bufferToImage(imageUpload.files[0])
        container.append(image)

        canvas = faceapi.createCanvasFromMedia(image)
        container.append(canvas)

        const displaySize = { width: image.width, height: image.height }
        faceapi.matchDimensions(canvas, displaySize)

        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()

        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        const results = resizedDetections.map((d) => {
            return faceMatcher.findBestMatch(d.descriptor)
        })
        results.forEach( (result, i) => {
            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
            drawBox.draw(canvas)
        })
    })
}

//load faces
function loadLabeledImages() {
    const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Tony Stark', 'Thor']
    loaderDiv.innerHTML = 'Loding Faces...'
    return Promise.all(
        labels.map(async (label)=>{
            const descriptions = []
            for(let i=1; i<=2; i++) {
                const img = await faceapi.fetchImage(`../labeled_images/${label}/${i}.jpg`)
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }
            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}