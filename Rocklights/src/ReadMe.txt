Parameters available to all states (prepend wth state name):

*LowTresh  
*LowTransToName 
*HighTresh 
*HighTransToName 

The LowThreash and HighThresh are whole number lower and upper population threshold
Below or above which the system enters the state specified by
LowTransToName and HighTransToName.  -1 means no threshold. The names can be comma
delimited lists.  (One name will be chose and random).  Or they can be
BACK which means go back to the last state.  (Becareful because the last
state might be an "intermission" 

*LifeTime
*TimeTransToName

Life time is a time based trigger to TimeTransToName in milisecs.  (1000 ms = 1 sec).
Name is either state, list, or BACK. -1 means no time trigger.

*Avatar 

See below for list of possiblites.  May be a comma delimited list.  Or NONE.  


*Ambient 

See below for list of possiblites.  May be a comma delimited list.  Or NONE.  


*TransSpeed

Controsl the speed in which this new state is intered into.  In milisecs  We never really
played with this.


Ambient: AmbEmpty - nonthing same as NONE
Ambient: AmbRedStick - the glowy red from before (in th empty room)
	has two settings RedStickDensityScalor and TrargetDensityScalor
	these are state specific (ie EmptyRedStickDensityScalor is differnt
	from FooRedStickDensityScalor).  See comments in profile.txt (state Empyt) 
	for details.	
Ambient: AmbBlueGreenSea - the blue green see from before

Ambient: GreenWave	-These all do the sonar type thing in differ colors
Ambient: BlueWave	-They have state (and color) specific parameters
Ambient: RedWave	-WaveMsPerStick and WaveWidth  
	(Eg SonarGreenWaveMsPerStick and FooRedWaveMsPerStick...
	WaveMsPerStick is the amount of time the wave spends on a stick before
	advancing.  The width is the total with of the wave (which is faded out)
Ambient: WillyWonka
Ambient: TargetTron - State specific parameters TronMSPerSquare controls the
	speed.	TronTailLength controls wether the trail shoudl be a fixed length or 
	spand the entire length between two targets -1

Ambient: KitRed			- withing a pixel lit pixel moves up and down
Ambient: KitRedCrazy	- for the non-crazy varients the whole room is
Ambient: KitBlue		- in sync.  For the crazy they are slightly out
Ambient: KitBlueCrazy   - of sync making for odd patters over time

Ambient: Sin           - blue taveling wave
Ambient: WashBang		- red lights travel up the walls
	then ceiling goes crazy with random colors.  State specific
	parameter WashSpeedScale.  (1.0 is about 8secs, .5 16 secs, 2 4 secs) 
Ambient: TargetFlash   - the target flash intermission from before
  -- has state specific parameter  TargetFlashHoldTime
Ambient: BGFlasher - very VERY brief random blue and green specs.  
    (for use with composer and a red ambient) 
Ambient: Composer - allows for the composition of multiple ambients with the state
	specific parameter Ambients

	

Available avatars:
Avatar: AV1Square - this is the normal avatar for before
Avatar: 9Square - this is the 9x9 avatar from before
Avatar: SonarPinger - when used with GreenWave will glow red when swepped over
					and play ping.wav
Avatar: AvHuge - this is the blue, green, randomized avatar from before
Avatar: SquareBUS - not sure think it has a animated blue trim around red center
Avatar: SinglePixelDance - not sure
Avatar: SquarePulse - not sure
Avatar: GENERIC - this is the biggy.  Allows for mixing and matching, 
enter patters, move patterns, and enter/exit/move sounds  see below for detials

GENERIC looks at the following parameters:
Patterns:
	OverheadPat 
	MovePat 
	EnterPat 
Availiable patterns are listed below.  Can be lists.  Can be NONE (or not specified)

Sounds:
	EnterSound 
	ExitSound 
	MoveSound 
Wave files to be triggers.  Files should be in the wavs directory.  No sound
should be specified with ""  (or just don't specify a sound for the state).

*EnterSoundLoop - specifes how many time the enter sound shoul be played.  
	the default is 1, -1 means loop until exit

*TailDelay - 0 means no tail.  Anything else and avatars have trails
*PillarMode - 1 floor to ceiling, -1 ceiling to floor, 0 no pillar

Patterns available to GENERIC avatar	

Pats: RedSquare - single red square
Pats: BlueSquare - single blue square
Pats: GreenSquare - single green square
Pats: RedSquareKnockOut  - the single squre but will knock out other overlappig colors
Pats: GreenSquareKnockOut- the single squre but will knock out other overlappig colors
Pats: BlueSquareKnockOut- the single squre but will knock out other overlappig colors
Pats: FlashBlock - flashes randomly for about .7 secs (for enter pattern for example)
Pats: TotalSquareKnockOut - single pixel subtracts color from any present
Pats: VelScale - use as move pattern creates 9x9 only when col or row changes (or enter pattern)
Pats: Plus - plus shape
Pats: Spinner - single red with a blue running around the outside
Pats: BlueLineKOH - blue horizontal line across the whole room (knocks out other color)
Pats: BlueLineKOV -blue verticle line across the whole room (knocks out other color)
Pats: O - red box with empty center
Pats: X - red x