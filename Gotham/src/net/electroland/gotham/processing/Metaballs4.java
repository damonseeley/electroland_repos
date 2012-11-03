package net.electroland.gotham.processing;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import net.electroland.elvis.net.GridData;
import net.electroland.gotham.processing.assets.Color;
import net.electroland.gotham.processing.assets.FastBlur;
import net.electroland.gotham.processing.assets.MetaballsProps;
import net.electroland.utils.ElectrolandProperties;

import org.apache.commons.collections15.buffer.BoundedFifoBuffer;

import processing.core.PVector;

@SuppressWarnings("serial")
public class Metaballs4 extends GothamPApplet {

    private List <MetaballGroup>groups;
    private GridData gridData;
    private MetaballsProps props;
    private BoundedFifoBuffer<GridData> gridHistory;
    private final static int HISTORY_LENGTH = 5;

    @Override
    public void setup(){

        String name = this.getProperties().getRequired("name");
        props = new MetaballsProps(this, name, new ElectrolandProperties("Gotham-global.properties"));

        // groups of balls
        groups = new ArrayList<MetaballGroup>();
        gridHistory = new BoundedFifoBuffer<GridData>(HISTORY_LENGTH);


        int redOrgRoam = 40; //was 100
        int purpRoam = 0; //was 0

        MetaballGroup red = new MetaballGroup(1, new Rectangle(-redOrgRoam, 
                                                            -redOrgRoam, 
                                                            this.getSyncArea().width + redOrgRoam, 
                                                            this.getSyncArea().height +redOrgRoam));
        groups.add(red);
        red.add(new Metaball(75));
        red.add(new Metaball(80));
        red.add(new Metaball(100));
        red.add(new Metaball(75));
        red.add(new Metaball(80));
        red.add(new Metaball(100));

        MetaballGroup orange = new MetaballGroup(2, new Rectangle(-redOrgRoam, 
                                                               -redOrgRoam, 
                                                               this.getSyncArea().width + redOrgRoam, 
                                                               this.getSyncArea().height +redOrgRoam));
        groups.add(orange);
        orange.add(new Metaball(70));
        orange.add(new Metaball(80));
        orange.add(new Metaball(90));
        orange.add(new Metaball(70));
        orange.add(new Metaball(80));

        MetaballGroup purple = new MetaballGroup(3, new Rectangle(purpRoam, 
                                                               purpRoam, 
                                                               this.getSyncArea().width + purpRoam, 
                                                               this.getSyncArea().height + purpRoam));
        groups.add(purple);
        purple.add(new Metaball(30));
        purple.add(new Metaball(40));
        purple.add(new Metaball(50));
        purple.add(new Metaball(50));
        purple.add(new Metaball(60));

        // probably should be in ball constructors
        for (MetaballGroup group : groups){
            for (Metaball ball : group.balls){
                ball.position  = new PVector(random(0, this.getSyncArea().width), random (0, this.getSyncArea().height));
                ball.velocity = new PVector(random(-1,1), random(-1,1));
            }
        }
    }

    @Override
    public void drawELUContent() {

        Rectangle gridCanvas = props.getBoundary(MetaballsProps.GRID_ON_CANVAS);
        boolean showGrid     = props.getState(MetaballsProps.SHOW_GRID);
        boolean enableGrid  = props.getState(MetaballsProps.ENABLE_GRID);

        // TODO: a lot of inefficiency here, getting values from the props file
        // in loops.

        // move balls
        for (MetaballGroup group : groups){

            PVector center = new PVector(0, 0, 0);

            for (Metaball ball : group.balls){

                boolean runningAway = false;

                if(!enableGrid){

                    if (mousePressed) {
                        ball.repell(new PVector(mouseX, mouseY, 0), props.getValue(MetaballsProps.REPELL_FORCE));
                        ball.spaceCheck(groups);
                        runningAway = true;
                    }

                } else if (gridData != null) {
                    synchronized(gridData){
                        List<Point> points = this.getObjects(gridData);
                        runningAway = points.size() > props.getValue("threshold");
                        for (Point point : points){

                            float cellWidth  = gridCanvas.width / (float)gridData.width;
                            float cellHeight = gridCanvas.height / (float)gridData.height;

                            PVector translated = new PVector((cellWidth * point.x) + gridCanvas.x, 
                                                             (cellHeight * point.y)  + gridCanvas.y);
                            // TODO: problem: we're applying force to the top left of each ball instead of
                            // it's center.  not so easy to fix: can't apply to translated vectors 
                            // because it would be per ball.  Can't apply to the balls easily because
                            // the ball object currently does boundary checks with the assumption that
                            // x,y is the top,left.
                            ball.repell(translated, props.getValue(MetaballsProps.REPELL_FORCE));
                        }
                    }
                }

                center.add(ball.position);
                ball.velocity.mult(props.getValue(MetaballsProps.FRICTION));
                // TODO: limit would work better as a coefficient.
                //  ball.velocity.limit(props.getValue(MetaballsProps.MAX_VELOCITY)) + num_balls * Coefficient
                //   should be calculated in head to avoid doing it over and over again.
                ball.velocity.limit(runningAway ? props.getValue(MetaballsProps.REPELL_VELOCITY_CEILING) : 
                                                  props.getValue(MetaballsProps.MAX_VELOCITY));

                if (ball.velocity.mag() < props.getValue(MetaballsProps.MIN_VELOCITY)){
                    ball.velocity.setMag(props.getValue(MetaballsProps.MIN_VELOCITY));
                }
            }

            center.div(group.balls.size());

            for (Metaball ball : group.balls){
                PVector c = PVector.sub(center, ball.position);
                c.normalize();
                c.mult(props.getValue(MetaballsProps.COHESIVENESS));
// disabled as a test case for keeping balls disperse.
                //                ball.velocity.add(c);
                ball.position.add(ball.velocity);
            }

            group.checkBounds();
        }

        // fill the whole area with purple
        Color bgColor = props.getColor(0);
        fill(bgColor.r, bgColor.g, bgColor.b, 127);
        rect(0, 0, width, height);

        // render each group's bounding box
        for (MetaballGroup group : groups){
            this.stroke(255,255,255);
            this.noFill();
            this.rect(group.range.x, group.range.y, group.range.width, group.range.height);
        }

        // render groups of balls
        for (MetaballGroup group : groups){
            Color color = props.getColor(group.id);
            float scale = props.getValue(MetaballsProps.BALL_SCALE);
            int ballOpacity = (int)props.getValue(MetaballsProps.BALL_OPACITY);

            for (Metaball ball : group.balls){
                this.noStroke();
                this.fill(color.r, color.g, color.b, ballOpacity);
                this.ellipse(ball.position.x, ball.position.y, ball.width() * scale, ball.height() * scale);
            }
        }

        // presence
        if (!enableGrid){
            if (mousePressed){
                fill(10, 200, 255);
                ellipse(mouseX, mouseY, 120, 120);
            }
        } else if (gridData != null) {

            fill(color(0, 0, 50), 8); //fill with a light alpha white
            rect(0, 0, this.getSyncArea().width, this.getSyncArea().height); //fill the whole area

            if (gridData != null){

                synchronized(gridData){

                    float cellWidth = gridCanvas.width / (float)gridData.width;
                    float cellHeight = gridCanvas.height / (float)gridData.height;

                    if (showGrid){
                        stroke(255);

                        for (int x = 0; x < gridData.width; x++){
                            for (int y = 0; y < gridData.height; y++){

                            	if (gridData.getValue(x, y) != (byte)0){
                                    fill(255);
                                }else{
                                    noFill();
                                }
                                this.rect(gridCanvas.x + (x * cellWidth), 
                                		  gridCanvas.y + (y * cellHeight), 
                                		  cellWidth, 
                                		  cellHeight);
                            }
                        }                    
                    
                    }else{

                    	noStroke();
                        fill(255, 255, 255, (int)props.getValue(MetaballsProps.PRESENCE_OPACITY));
                        float presenceRadius = props.getValue(MetaballsProps.PRESENCE_RADIUS);

                        for (GridData data : gridHistory){
                        	for (int x = 0; x < data.width; x++){
                                for (int y = 0; y < data.height; y++){
                                    if (data.getValue(x, y) != (byte)0){
                                        this.ellipse(gridCanvas.x + (x * cellWidth), 
                                                     gridCanvas.y + (y * cellHeight), 
                                                     presenceRadius, presenceRadius);
                                    }
                                }
                            }                         	
                        }
                                      
                    }
                }
            }
        }

        if (!showGrid){
            loadPixels();
            FastBlur.performBlur(pixels, width, height, floor(props.getValue(MetaballsProps.BLUR)));
            updatePixels();
        }
    }

    public List<Point> getObjects(GridData grid){
        List<Point>objects = new ArrayList<Point>();
        for (int i = 0; i < grid.data.length; i++){
            if (grid.data[i] != (byte)0){
                int y = i / grid.width;
                int x = i - (y * grid.width);
                objects.add(new Point(x,y));
            }
        }
        return objects;
    }


    @Override
    public void handle(GridData srcData) {

        if (gridData == null){
            gridData = srcData;
        } else {
            synchronized(gridData){

                // copy the original source, so we don't accidentally change
                // the source for other clients using this data.
                StringBuilder sb = new StringBuilder();
                srcData.buildString(sb);
                srcData = new GridData(sb.toString());

                srcData = subset(srcData, props.getBoundary(MetaballsProps.GRID));

                srcData = counterClockwise(srcData);

                if (props.getState(MetaballsProps.MIRROR_HORIZONTAL)){
                    srcData = flipHorizontal(srcData);
                }
                if (props.getState(MetaballsProps.MIRROR_VERTICAL)){
                    srcData = flipVertical(srcData);
                }
                gridData = srcData;
                if (gridHistory.isFull()){ gridHistory.remove(); }
                gridHistory.add(gridData);
            }
        }
    }

    // unit test for grid operations
    public static void main(String args[]){
        GridData data = new GridData("6,5,0,1,2,3,4,5,0,0,0,0,0,0,0,1,2,3,4,5,1,1,1,1,1,1,0,1,2,3,4,5");
        printGrid(data);
        data = subset(data, new Rectangle(2,1,3,4));
        printGrid(data);
        data = counterClockwise(data);
        printGrid(data);
        data = counterClockwise(data);
        printGrid(data);
        data = flipVertical(data);
        printGrid(data);
        data = flipHorizontal(data);
        printGrid(data);
    }

    public static void printGrid(GridData grid){
        for (int y = 0; y < grid.height; y++){
            for (int x = 0; x < grid.width; x++){
                System.out.print(grid.data[y * grid.width + x]);
            }
            System.out.println();
        }
        System.out.println();
    }


    /**
     * returns new GridData where data is a subset of the original based on
     * a Rectangular boundary.
     * 
     * @param in
     * @param boundary
     * @return
     */
    public static GridData subset(GridData in, Rectangle boundary){

        try {
			byte[] target = new byte[boundary.width * boundary.height];
			for (int y = 0; y < boundary.height; y++) {
			    try {
			        System.arraycopy(in.data, ((y + boundary.y) * in.width) + (boundary.x), target, y * boundary.width, boundary.width);
			    } catch (Exception e) {
			        e.printStackTrace();
			    }
			}
			in.data  = target;
			in.height = boundary.height;
			in.width  = boundary.width;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        return in;
    }

    public static GridData counterClockwise(GridData in){

        // switch height for width and vice versa
        byte[] rotated = new byte[in.data.length];
        int i = 0;

        for (int x = in.width - 1; x >= 0; x--){
            for (int y = 0; y < in.height; y++){
                rotated[i++] = in.data[y * in.width + x];
            }
        }
        int w = in.width;
        in.width = in.height;
        in.height = w;
        in.data = rotated;

        return in;
    }

    public static GridData flipVertical(GridData in){
        byte[] flipped = new byte[in.data.length];

        for (int y = 0; y < in.height; y++){
            System.arraycopy(in.data, y * in.width, flipped, flipped.length - ((y + 1) * in.width), in.width);
        }
        in.data = flipped;
        return in;
    }

    public static GridData flipHorizontal(GridData in){
        int center = in.width / 2;
        byte buffer;

        for (int y = 0; y < in.height; y++){
            for (int x = 0; x < center; x++){
                int leftIndex = y * in.width + x;
                int rightIndex = (y + 1) * in.width - x - 1;
                buffer = in.data[leftIndex];
                in.data[leftIndex] = in.data[rightIndex];
                in.data[rightIndex] = buffer;
            }
        }
        return in;
    }

    public void consoleDump(){
    	for (MetaballGroup group : groups){
    		for (Metaball ball : group.balls){
    			System.out.println(ball);
    		}
    	}
    }
    
    class MetaballGroup {

        Rectangle range;
        int id;
        List <Metaball>balls;

        public MetaballGroup(int id, Rectangle range){
            this.range = range;
            this.id = id;
            balls = new ArrayList<Metaball>();
        }

        public void add(Metaball ball){
            ball.group = this;
            balls.add(ball);
        }
        
        public void checkBounds(){
            for (Metaball ball : balls){
                ball.checkBounds(range);
            }
        }
    }

    class Metaball {

        float radius;
        PVector position;
        PVector velocity;
        MetaballGroup group;

        public Metaball(float radius){
            this.radius = radius;
        }

        public String toString(){
        	StringBuffer br = new StringBuffer("Metaball[");
        	br.append("group=").append(group.id).append(", ");
        	br.append("radius=").append(radius).append(", ");
        	br.append("position=").append(position).append(", ");
        	br.append("velocity=").append(velocity).append(", ");
        	br.append("color=").append(props.getColor(group.id));
        	br.append(']');
        	return br.toString();
        }
        
        public void checkBounds(Rectangle range){
            // bounce
            if (left() < range.x){
                setLeft(range.x);
                velocity.x = -velocity.x;
            } else if (right() > range.width + range.x){
                setRight(range.width + range.x);
                velocity.x = -velocity.x;
            }
            if (top() < range.y){
                setTop(range.y);
                velocity.y = -velocity.y;
            } else if (bottom() > range.height + range.y){
                setBottom(range.height + range.y);
                velocity.y = -velocity.y;
            }
        }

        public void repell(PVector point, float force){

            float mDistance = PVector.dist(position, point);
            if (mDistance > 0){ // potential divide-by-zero
                PVector repel = PVector.sub(position, point);
                repel.normalize();
                repel.mult(force / (mDistance * mDistance));
                velocity.add(repel);
            }else{
                // what's a proper resolution here?
                System.err.println("divide by zero in repell.");
                System.out.println("and velcity is left at ..." + velocity);
            }
        }

        public void spaceCheck(List<MetaballGroup> groups){
            for (MetaballGroup group : groups) {
                if (group != this.group) {
                    for (Metaball ball : group.balls){
                        if (ball.overlaps(this) || ball.group != this.group){
                            this.repell(ball.position, Metaballs2.BALL_REPELL_FORCE);
                        }
                    }
                }
            }
        }

        public boolean overlaps(Metaball ball){
            float distance = ball.position.dist(this.position);
            return distance < ball.radius || distance < this.radius;
        }

        public float width(){
            return radius;
        }
        public float height(){
            return radius;
        }

        public float left(){
            return position.x - width() / 2;
        }
        public float right(){
            return position.x + width() / 2;
        }
        public float top(){
            return position.y - height() / 2;
        }
        public float bottom(){
            return position.y + height() / 2;
        }

        public void setLeft(float left){
            position.x = width() / 2 + left; 
        }
        public void setRight(float right){
            position.x = right - width() / 2;
        }
        public void setTop(float top){
            position.y = top + height() / 2;
        }
        public void setBottom(float bottom){
            position.y = bottom - height() / 2;
        }
    }
}