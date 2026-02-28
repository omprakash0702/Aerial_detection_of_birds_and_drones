`let stream;
let running=false;

function clearOutput(){
resultImg.src=""
resultVideo.src=""
liveResult.src=""
resultText.innerHTML=""
}

imageInput.onchange=e=>{
clearOutput()
resultImg.src=URL.createObjectURL(e.target.files[0])
}

videoInput.onchange=e=>{
clearOutput()
resultText.innerText="🎥 Video Ready"
}

async function uploadImage(){
clearOutput()
resultText.innerText="Processing..."

let file=imageInput.files[0]
let mode=document.getElementById("mode").value

let formData=new FormData()
formData.append("file",file)
formData.append("mode",mode)

let response=await fetch("/predict-image",{method:"POST",body:formData})

if(mode=="classification"){
let data=await response.json()
resultText.innerHTML=
(data.label=="Bird"?"🐦 Bird":"🤖 Drone")+
" ("+(data.confidence*100).toFixed(2)+"%)"
}else{
let blob=await response.blob()
resultImg.src=URL.createObjectURL(blob)
resultText.innerText="🎯 Detection Complete"
}
}

async function uploadVideo(){
clearOutput()
resultText.innerText="Processing Video..."

let file=videoInput.files[0]
let formData=new FormData()
formData.append("file",file)

let response=await fetch("/predict-video",{method:"POST",body:formData})
let blob=await response.blob()

resultVideo.src=URL.createObjectURL(blob)
resultVideo.load()
resultVideo.play()

resultText.innerText="🎥 Done"
}

function startServerCam(){
clearOutput()
resultImg.src="/server-webcam"
}

async function startPhoneCam(){
clearOutput()
stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"}})
camera.srcObject=stream
running=true
detectLoop()
}

function stopPhoneCam(){
running=false
if(stream){
stream.getTracks().forEach(track=>track.stop())
}
}

async function detectLoop(){
if(!running)return
let canvas=document.createElement("canvas")
canvas.width=camera.videoWidth
canvas.height=camera.videoHeight
let ctx=canvas.getContext("2d")
ctx.drawImage(camera,0,0)

canvas.toBlob(async blob=>{
let formData=new FormData()
formData.append("file",blob,"frame.jpg")
let res=await fetch("/predict-frame",{method:"POST",body:formData})
let imgBlob=await res.blob()
liveResult.src=URL.createObjectURL(imgBlob)
},"image/jpeg")

setTimeout(detectLoop,500)
}`
let stream;
let running=false;

function clearOutput(){
    resultImg.src=""
    resultVideo.src=""
    liveResult.src=""
    resultText.innerHTML=""
}

imageInput.onchange=e=>{
    clearOutput()
    resultImg.src=URL.createObjectURL(e.target.files[0])
}

videoInput.onchange=e=>{
    clearOutput()
    resultText.innerText="🎥 Video Ready"
}

async function uploadImage(){
    clearOutput()
    resultText.innerText="Processing..."

    let file=imageInput.files[0]
    let mode=document.getElementById("mode").value

    let formData=new FormData()
    formData.append("file",file)
    formData.append("mode",mode)

    let response=await fetch(
        window.location.origin + "/predict-image",
        {method:"POST",body:formData}
    )

    if(mode=="classification"){
        let data=await response.json()
        resultText.innerHTML=
        (data.label=="Bird"?"🐦 Bird":"🤖 Drone")+
        " ("+(data.confidence*100).toFixed(2)+"%)"
    }else{
        let blob=await response.blob()
        resultImg.src=URL.createObjectURL(blob)
        resultText.innerText="🎯 Detection Complete"
    }
}

async function uploadVideo(){
    clearOutput()
    resultText.innerText="Processing Video..."

    let file=videoInput.files[0]
    let formData=new FormData()
    formData.append("file",file)

    let response=await fetch(
        window.location.origin + "/predict-video",
        {method:"POST",body:formData}
    )

    let blob=await response.blob()

    resultVideo.pause()
    resultVideo.src=""
    resultVideo.src=URL.createObjectURL(blob)
    resultVideo.load()
    resultVideo.play()

    resultText.innerText="🎥 Done"
}

async function startPhoneCam(){
    clearOutput()
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"}})
    camera.srcObject=stream
    running=true
    detectLoop()
}

function stopPhoneCam(){
    running=false
    if(stream){
        stream.getTracks().forEach(track=>track.stop())
    }
}

async function detectLoop(){
    if(!running) return;

    let canvas=document.createElement("canvas")
    canvas.width=camera.videoWidth
    canvas.height=camera.videoHeight
    let ctx=canvas.getContext("2d")
    ctx.drawImage(camera,0,0)

    let blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"))

    let formData=new FormData()
    formData.append("file",blob,"frame.jpg")

    let res=await fetch(
        window.location.origin + "/predict-frame",
        {method:"POST",body:formData}
    )

    let imgBlob=await res.blob()
    liveResult.src=URL.createObjectURL(imgBlob)

    requestAnimationFrame(detectLoop)
}