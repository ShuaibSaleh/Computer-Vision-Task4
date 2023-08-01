let imageA_input = document.querySelector('#imageA_input')
let imageA = document.querySelector('#imageA')
let submit = document.getElementById("submit_btn")
let imageA_data = ""

// let operation = document.getElementById("operation")
// let subOperation = document.getElementById("sub-operation")


const operation = document.getElementById('operation');
const subOperation = document.getElementById('sub-operation');

  operation.addEventListener('change', () => {
    const selectedOperation = operation.value;
    const subOperationGroups = subOperation.getElementsByTagName('optgroup');
    
    for (const group of subOperationGroups) {
      if (group.label.toLowerCase() === selectedOperation.toLowerCase()) {
        group.style.display = 'block';
      } else {
        group.style.display = 'none';
      }
    }
  });


imageA_input.addEventListener('change', e => {
    if (e.target.files.length) {
        // start file reader
      const reader = new FileReader();
      reader.onload = e => {
        if (e.target.result) {
          // create new image
          let img = document.createElement('img');
          img.id = 'imageA';
          img.src = e.target.result;
          // clean result before
          imageA.innerHTML = '';
          // append new image
          imageA.appendChild(img)
          // origial image
          imageA_data = e.target.result

        }
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });



function send(){
  
    // to handle if the user not enter two images
    try {
      if (imageA_data == "" ) {
        throw "error : not enought images "
      }

      let formData = new FormData();
      formData.append('imageA_data',imageA_data)
      formData.append('operation',operation.value)
      formData.append('subOperation',subOperation.value)
      
      
    
      $.ajax({
        type: 'POST',
        url: '/processing',
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
        async: true,
        success: function (backEndData) {
          var responce = JSON.parse(backEndData)

          let imageC = document.getElementById("imageC")
          imageC.remove()
          imageC = document.createElement("div")
          imageC.id = "imageC"
          imageC.innerHTML = responce[1]
          
          let imagediv3 = document.getElementById("imagediv3")
          imagediv3.appendChild(imageC)
          
  
        }
  
  
      })
    } catch (error) {
      console.log("please upload two images")
    } 
  }


submit.addEventListener('click', e => {
    e.preventDefault();
    send()
  }
  )