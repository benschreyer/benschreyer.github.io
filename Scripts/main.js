var c = document.getElementById("canvas").getContext("2d");
var background=c.createPattern(document.getElementById("background"),"repeat");
var pup=c.createPattern(document.getElementById("pup"),"repeat");
var pdown=c.createPattern(document.getElementById("pdown"),"repeat");
var pleft=c.createPattern(document.getElementById("pleft"),"repeat");
var pright=c.createPattern(document.getElementById("pright"),"repeat");
c.fillStyle=background;
c.fillRect(0,0,500,500);
var player=new player();
document.addEventListener("keypress", function(event)
			{switch(event.which)
					{
					case 119:
						console.log("press");
						player.velocity[0]=0;
						player.velocity[1]=-2;
						break;
					case 97:
						console.log("press");
						player.velocity[0]=-2;
						player.velocity[1]=0;
						break;
					case 115:
						console.log("press");
						player.velocity[0]=0;
						player.velocity[1]=2;
						break;
					case 100:
						console.log("press");
						player.velocity[0]=2;
						player.velocity[1]=0;
						break;
					}});
function loop()
	{
	c.fillStyle=background;
	c.fillRect(0,0,500,500);
	player.update();
	}
setInterval(loop,25);
