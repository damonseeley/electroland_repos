package net.electroland.duke {
	
	import flash.display.MovieClip;
	
	public class DukePrototypeMain extends MovieClip{
		
		private var particleSystem:ParticleSystem;
		private var controlPanel:ControlPanel;
		private var macroManager:MacroManager;
		
		public function DukePrototypeMain(){
			particleSystem = new ParticleSystem();
			addChild(particleSystem);
			particleSystem.setup(20, 1);
			controlPanel = new ControlPanel(particleSystem);
			addChild(controlPanel);
			macroManager = new MacroManager(particleSystem, controlPanel);
			controlPanel.setup(macroManager);
			particleSystem.addControlPanel(controlPanel);
			addChild(macroManager);
			macroManager.start();
		}
		
	}
	
}