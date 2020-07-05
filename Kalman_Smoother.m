%EGR 680:Advanced Controls: Project 3- Part1-Extra Credit
%Name: Prashant Adhikari
%Professor: Dr. Nicholas Baine
%Objective: To apply Kalman Smoother to a two-dimensional Brownian random
%process to derive an optimal estimate.
%-----------------------------------------------------------------------------------------------------

load('Project3Part1.mat'); %Importing x_true -> truth measurement and z -> given measurement
F = [1 0;0 1];
H = [1 0;0 1];
R = [0.1 0;0 0.1];
Q = [0.1^2 0;0 0.1^2];
%-----------------------------------------------------------------------------------------------------
%Initializing forward pass
x_pred_forward = [0;0]; %X0|0
P_pred_forward = [10^3 0;0 10^3];%P0|0 
x_upd_forward = [];
P_upd_forward = [];
%-----------------------------------------------------------------------------------------------------
%Initializing smoother 
x_upd_smoother = [];
%-----------------------------------------------------------------------------------------------------
%Implementing forward pass discrete Kalman filter
for n = 1:1:length(x_true)
    x_pred_forward = F*x_pred_forward; %predicted (a priori) state estimate
    P_pred_forward = F*P_pred_forward*F'+Q;%predicted (a priori) estimate of error covariance
    y_forward = z(:,n)-H*x_pred_forward;% innovation (measurement residual)
    S_forward = H*P_pred_forward*H'+R;%innovation/measurement covariance
    K_forward = P_pred_forward*H'*S_forward^-1; %optimal Kalman gain  
    x_pred_forward = x_pred_forward+K_forward*y_forward;%updated (a posteriori) estimate at state estimate covariance
    x_upd_forward = [x_upd_forward,x_pred_forward];
    P_pred_forward = (eye(2)-K_forward*H)*P_pred_forward;%updated (a  posteriori) state estimate
    P_upd_forward = [P_upd_forward,P_pred_forward];
    trace_Pred_forward(n) = sqrt(trace(P_pred_forward));% trace of estimated error covariance matrix
end
%-----------------------------------------------------------------------------------------------------
%Initializing backward pass
x_pred_backward = [0;0]; %X0|0
P_pred_backward = [10^3 0;0 10^3]; %P0|0
x_upd_backward = [];
P_upd_backward = [];
%Implementing backward pass discrete Kalman filter
j = 1:2:2000;

for i = length(x_true):-1:1
    x_pred_backward = F^-1*x_pred_backward;%predicted (a priori) state estimate
    P_pred_backward = F^-1*P_pred_backward*(F^-1)'+Q;%predicted (a priori) estimate of error covariance
    y_backward = z(:,i)-H*x_pred_backward;% innovation (measurement residual)
    S_backward = H*P_pred_backward*H'+R;%innovation/measurement covariance
    K_backward = P_pred_backward*H'*S_backward^-1; %optimal Kalman gain 
    x_pred_backward = x_pred_backward+K_backward*y_backward;%updated (a posteriori) estimate at state estimate covariance
    x_upd_backward = [x_upd_backward,x_pred_backward];
    P_pred_backward = (eye(2)-K_backward*H)*P_pred_backward;%updated (a  posteriori) state estimate   
    P_upd_backward = [P_upd_backward,P_pred_backward];
    trace_Pred_backward(i) = sqrt(trace(P_pred_backward));% trace of estimated error covariance matrix
    
    x_pred_smoother = x_upd_forward(:,i)+P_upd_forward(:,j(i):j(i)+1)*(P_upd_forward(:,j(i):j(i)+1)+P_pred_backward)^-1*(x_pred_backward-x_upd_forward(:,i));
    x_upd_smoother=[x_upd_smoother,x_pred_smoother];
    P_pred_smoother = (P_upd_forward(:,j(i):j(i)+1)*(P_upd_forward(:,j(i):j(i)+1)+P_pred_backward)^-1)*P_pred_backward;
    trace_Pred_smoother(i) = sqrt(trace(P_pred_smoother));
   
end

%-----------------------------------------------------------------------------------------------------
%Root Mean Square Error calculation for forward, backward and smoother
for k = 1:length(x_true)
    RMSE_smoother_state1(k) = sqrt(sum((x_upd_smoother(1,1:k)-x_true(1,1:k)).^2))/k;
    RMSE_smoother_state2(k) = sqrt(sum((x_upd_smoother(2,1:k)-x_true(2,1:k)).^2))/k;
    RMSE_forward_state1(k) = sqrt(sum((x_upd_forward(1,1:k)-x_true(1,1:k)).^2))/k;
    RMSE_forward_state2(k) = sqrt(sum((x_upd_forward(2,1:k)-x_true(2,1:k)).^2))/k;
    RMSE_backward_state1(k) = sqrt(sum((x_upd_backward(1,1:k)-x_true(1,1:k)).^2))/k;
    RMSE_backward_state2(k) = sqrt(sum((x_upd_backward(2,1:k)-x_true(2,1:k)).^2))/k;
end
% %-----------------------------------------------------------------------------------------------------
%Plotting trace for forward, backward and smoother
 n1 = 1:1:length(x_true);
subplot(3,1,1);
plot(n1,trace_Pred_forward,'r','Linewidth',2);
hold on;
plot(n1,trace_Pred_backward,'b','Linewidth',2);
hold on;
plot(n1,trace_Pred_smoother,'k','Linewidth',2);
xlabel('n');
ylabel('trace(P)');
title('Trace of estimated error covariance matrix ');
legend('forward pass','backward pass','smoother');
%-----------------------------------------------------------------------------------------------------
subplot(3,1,2);
plot(n1,x_true(1,:),'b','Linewidth',2);
hold on;
plot(n1,x_upd_smoother(1,:),'k','Linewidth',2);
hold on;
xlabel('n');
ylabel('Values');
title('Estimate vs True Values for state 1');
legend('True','smoother');
%-----------------------------------------------------------------------------------------------------
subplot(3,1,3);
plot(n1,x_true(2,:),'b','Linewidth',2);
hold on;
plot(n1,x_upd_smoother(2,:),'k','Linewidth',2);
hold on;
xlabel('n');
ylabel('Values');
title('Estimate vs True Values for state 2');
legend('True','smoother');
    
    