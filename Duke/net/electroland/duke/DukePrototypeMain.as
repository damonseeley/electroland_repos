package net.electroland.duke {
	
	import flash.display.MovieClip;
	
	public class DukePrototypeMain extends MovieClip{
		
		private var particleSystem:ParticleSystem;
		private var controlPanel:ControlPanel;
		
		public function DukePrototypeMain(){
			particleSystem = new ParticleSystem();
			addChild(particleSystem);
			particleSystem.setup(50, 3);
			controlPanel = new ControlPanel(particleSystem);
			particleSystem.addControlPanel(controlPanel);
			addChild(controlPanel);
		}
		
	}
	
}