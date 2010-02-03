package net.electroland.duke {
	
	import flash.display.MovieClip;
	
	public class DukePrototypeMain extends MovieClip{
		
		private var particleSystem:ParticleSystem;
		
		public function DukePrototypeMain(){
			particleSystem = new ParticleSystem();
			addChild(particleSystem);
			particleSystem.setup(50, 3);
		}
		
	}
	
}