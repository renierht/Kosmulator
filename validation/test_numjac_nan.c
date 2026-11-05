/* Direct guard regression: invalid derivatives must never index Rowmax. */
#include "evolver_ndf15.h"
#include <string.h>
int main(void) {
  struct jacobian jac = {0};
  struct numjac_workspace ws = {0};
  double y[2] = {0., 1.}, f[2] = {0., NAN};
  int calls=0;
  ErrorMsg error;
  int status=numjac(NULL,-50.,y,f,&jac,&ws,1.e-15,1,&calls,NULL,error);
  if (status != _FAILURE_ || strstr(error,"Non-finite state or derivative") == NULL) return 1;
  puts(error);
  return 0;
}
