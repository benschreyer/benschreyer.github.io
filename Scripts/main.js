
var c = document.getElementById("canvas").getContext("2d");
var background=c.createPattern(document.getElementById("background"),"repeat");
var pup=c.createPattern(document.getElementById("pup"),"repeat");
var pdown=c.createPattern(document.getElementById("pdown"),"repeat");
var pleft=c.createPattern(document.getElementById("pleft"),"repeat");
var pright=c.createPattern(document.getElementById("pright"),"repeat");
c.fillStyle=background;
c.fillRect(0,0,500,500);
c.fillStyle=pup;
c.fillRect(0,0,50,50);
c.fillStyle=pleft;
c.fillRect(0,100,50,50);
