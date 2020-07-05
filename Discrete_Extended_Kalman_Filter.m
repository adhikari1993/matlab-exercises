%EGR 680:Advanced Controls: Project 3- Part1
%Name: Prashant Adhikari
%Professor: Dr. Nicholas Baine
%Objective: To apply Kalman filter to a two-dimensional Brownian random
%process to derive an optimal estimate.
%-----------------------------------------------------------------------------------------------------

load('Project3Part1.mat'); %Importing x_true -> truth measurement and z -> given measurement
%-----------------------------------------------------------------------------------------------------
% Given
F=[1 0;0 1]; % Fk
H=[1 0;0 1]; % Hk
R=[1 0;0 1]; % Rk
Q=[0.1^2 0;0 0.1^2]; % Qk
%-----------------------------------------------------------------------------------------------------
%Initialization
P_predict=[10^3 0;0 10^3]; %P0|0
X_predict=[0;0]; %X0|0
X_update=[];
P_update=[];
error=zeros(2,length(x_true));
%-----------------------------------------------------------------------------------------------------
%Implementing discrete Kalman Filter
for n=1:length(x_true)
    X_predict=F*X_predict; %predicted (a priori) state estimate
    P_predict=F*P_predict*F'+Q; %predicted (a priori) estimate of error covariance
    Y=z(:,n)-H*X_predict; % innovation (measurement residual)
    S=H*P_predict*H'+R; %innovation/measurement covariance
    K=P_predict*H'*S^-1; %optimal Kalman gain
    P_predict=(eye(2)-K*H)*P_predict; %updated (a posteriori) estimate at state estimate covariance
    X_predict=X_predict + K*Y; %updated (a  posteriori) state estimate
    X_update=[X_update,X_predict];
    P_update=[P_update,P_predict];
    traceP(n)=sqrt(trace(P_predict));% trace of estimated error covariance matrix
    %error(:,n)=[(x_true(1,n)-X_update(1,n))^2;(x_true(2,n)-X_update(2,n))^2]; %error computation for state1 and state2
end
%-----------------------------------------------------------------------------------------------------
%plotting true and estimate graphs for state 1 and state 2
n=1:1:length(x_true);
subplot(2,2,1);
plot(n,x_true(1,:),'Linewidth',2);
hold on;
plot(n,X_update(1,:),'Linewidth',2);
hold on;
grid on;
legend('True', 'Estimate');
title('Estimate values vs True values for state 1');
xlabel('n');
ylabel('Value');


subplot(2,2,2);
plot(n,x_true(2,:),'Linewidth',2);
hold on;
plot(n,X_update(2,:),'Linewidth',2);
hold on;
grid on;
legend('True', 'Estimate');
title('Estimate values vs True values for state 2');
xlabel('n');
ylabel('Value');
% %-----------------------------------------------------------------------------------------------------
%Root mean square error calculation for state 1 and state 2
for i = 1:length(x_true)
    RMSE_State1(i) = sqrt(sum((X_update(1,1:i)-x_true(1,1:i)).^2))/i;
end

for i = 1:length(x_true)
    RMSE_State2(i) = sqrt(sum((X_update(2,1:i)-x_true(2,1:i)).^2))/i;
end
%-----------------------------------------------------------------------------------------------------
%plotting root mean square error graphs for state 1 and state 2
subplot(2,2,3);
plot(n,RMSE_State1,'Linewidth',2);
hold on;
plot(n,RMSE_State2,'Linewidth',2);
grid on;
legend('State 1', 'State 2');
title('Root mean square error');
xlabel('n');
ylabel('RMSE');
%-----------------------------------------------------------------------------------------------------
%RMSE = sqrt(sum(error,2)/1000) %Root mean square error values for state 1 and state 2
%-----------------------------------------------------------------------------------------------------
% Plotting trace of estimate errror covariance matrix
subplot(2,2,4);
plot(n,traceP,'Linewidth',2);
title('Trace of estimated error covariance matrix');
xlabel('n');
ylabel('trace(P)');

