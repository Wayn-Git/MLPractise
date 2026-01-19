/* Method Overriding:It is a OOP feature where the subclass has a different implimentation 
or we can say output while preserving the other aspects of the method/methods in the parent class or the class it's 
Inheritating from. 

Rules FOr Method Overriding are:
- The parameters should be same
- The Method name should be the same (Ofc we are overrding it)
- Shouldn't be a final method/class
- (Optional) Using @Override annotation to help catch errors and better understanding
- Sub Class method can be public or protected but not private
- Static method cannot be overriden 
- Parent class cannot be a private class
*/
class Shape {
    double area() {
        return 0;
    }
}

class Rectangle extends Shape {
    @Override
    double area() {
        return 10 * 5;
    }
}

class Circle extends Shape {
    @Override
    double area() {
        return 3.14 * 7 * 7;
    }
}

class Main {
    public static final String ANSI_RESET = "\u001B[0m";

    public static final String BLACK = "\u001B[30m";

    public static final String WHITE_BACKGROUND = "\u001B[47m";


    public static void main(String[] args) {
        Shape s1 = new Rectangle();
        Shape s2 = new Circle();

         System.out.println(WHITE_BACKGROUND + BLACK + "Output: " + ANSI_RESET);

        System.out.println("Area of Rectangle: " + s1.area());
        System.out.println();
        System.out.println("Area of Circle: " + s2.area());
    }
}
