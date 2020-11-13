/*
The standard ASCII table defines 128 character codes (from 0 to 127), of
which, the first 32 are control codes (non-printable), and the remaining 96
character codes are representable characters:
*/
 
#include <iostream>
#include <iomanip>
using namespace std;
 
int main()
{   int i,j;
    char cmd[32][4]= {"NUL","SOH","STX","ETX","EOT","ENQ","ACK","BEL","BS","TAB",
                      "LF","VT","FF","CR","SO","SI","DLE","DC1","DC2","DC3","DC4","NAK",
                      "SYN","ETB","CAN","EM","SUB","ESC","FS","GS","RS","US"};
     cout << "The standard ASCII table defines 128 character codes (from 0 to 127),";
     cout << "\n of which,the first 32 are control codes (non-printable), and the";
     cout << "\n remaining 96 charactercodes are representable characters:\n";
        cout << "*";
    for( i = 0 ; i < 10 ; i++)
        cout << setw(4) << i;
 
    for( i = 0x41 ; i < 0x47 ; i++)
        cout << setw(4) << static_cast<char>(i);
        cout << endl << "-- ";
 
    for( i = 0 ; i < 16 ; i++)
     cout << left << "--- ";
 
    for( i = 0 ; i < 2 ; i++ )
        {
            cout << endl <<  i << "| " ;
 
    for(j = 0 ; j < 16 ; j++)
        cout << setw(4) << left << cmd[i*16+j];
        }
    for( i = 2 ; i < 8 ; i++ )
        {
        cout << endl << i << "| " ;
    for( j = 0 ; j < 16 ; j++)
            if((i*16 + j) != 127 )
         cout << setw(4) << left << static_cast<char>(i*16+j);
        }
    cout << endl << endl;
    return 0;
}
/*Program's output
The standard ASCII table defines 128 character codes (from 0 to 127),
 of which,the first 32 are control codes (non-printable), and the
 remaining 96 charactercodes are representable characters:
*   0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
0| NUL SOH STX ETX EOT ENQ ACK BEL BS  TAB LF  VT  FF  CR  SO  SI
1| DLE DC1 DC2 DC3 DC4 NAK SYN ETB CAN EM  SUB ESC FS  GS  RS  US
2|     !   "   #   $   %   &   '   (   )   *   +   ,   -   .   /
3| 0   1   2   3   4   5   6   7   8   9   :   ;   <   =   >   ?
4| @   A   B   C   D   E   F   G   H   I   J   K   L   M   N   O
5| P   Q   R   S   T   U   V   W   X   Y   Z   [   \   ]   ^   _
6| `   a   b   c   d   e   f   g   h   i   j   k   l   m   n   o
7| p   q   r   s   t   u   v   w   x   y   z   {   |   }   ~
 
 
Process returned 0 (0x0)   execution time : 0.078 s
Press any key to continue.
 
*/
