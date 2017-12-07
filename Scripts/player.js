function player()
	{
		this.vector=[0,0];
		this.velocity=[2,0];
		this.update=function()
			{
				this.vector[0]+=this.velocity[0];
				this.vector[1]+=this.velocity[1];
				switch(this.velocity[0])
					{
					case 2:
						c.fillStyle=pright;
						break;
					case -2:
						c.fillStyle=pleft;
						break;
					case 0:
						switch(this.velocity[1])
							{
							case 2:
								c.fillStyle=pdown;
								break;
							case -2:
								c.fillStyle=pup;
								break;
							}
						break;
					}
				c.translate(this.vector[0],this.vector[1]);
				c.fillRect(0,0,50,50);
				c.translate(-1*this.vector[0],-1*this.vector[1]);
			}
		this.init=function()
			{
			
			}

	}
		
