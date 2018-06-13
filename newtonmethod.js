function f(x) {
 return Math.pow(x,6)*5+Math.pow(x,4)*2-Math.pow(x,3)*12+Math.pow(x,2)*3;
}
function fd(x){
return Math.pow(x,5)*30+Math.pow(x,3)*8-Math.pow(x,2)*36+Math.pow(x,1)*6;
}
var guess=1200;
while(f(guess)>.0001 || f(guess)<-.0001)
{
  guess=guess-(f(guess))/(fd(guess));
}
console.log(guess);
