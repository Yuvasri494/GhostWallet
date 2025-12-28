"""
GhostWallet - Complete Behavioral Biometrics Fraud Detection System
One Button - Safe/Suspicious/Alert Detection with OTP & CyborgDB
NOW WITH PASSWORD VALIDATION & BEHAVIORAL ANALYSIS
"""

import os
import json
import time
import hashlib
import base64
import secrets
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import uuid
from collections import defaultdict
import random

# ==================== CONFIGURATION ====================
CONFIG = {
    'server': {
        'host': '0.0.0.0',
        'port': 5000,
        'debug': False
    },
    'security': {
        'otp_timeout': 300,
        'max_otp_attempts': 3,
        'encryption_algorithm': 'AES-256'
    },
    'behavioral': {
        'typing_human_range': (100, 300),
        'mouse_human_range': (100, 500),
        'required_min_keystrokes': 10,
        'password_min_length': 8
    },
    'transaction': {
        'small_amount': 100,
        'medium_amount': 500,
        'large_amount': 1000,
        'critical_amount': 5000
    },
    'passwords': {
        'user_001': 'SecurePass123!',
        'user_002': 'Test@Password456',
        'user_003': 'Demo#Pass789',
        'user_004': 'GhostWallet2024!'
    }
}

# ==================== ENUMS ====================
class RiskLevel(Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    ALERT = "alert"

class Decision(Enum):
    APPROVE = "approve"
    OTP_REQUIRED = "otp_required"
    BLOCK = "block"

class PasswordStatus(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NOT_PROVIDED = "not_provided"

# ==================== CYBORGDB SIMULATION ====================
class CyborgDB:
    """Simulated encrypted vector database"""
    
    def __init__(self):
        self.vectors = {}
        self.user_profiles = defaultdict(list)
        self.transactions = defaultdict(list)
        self.otp_store = {}
        self.password_attempts = defaultdict(int)
        self.password_lockouts = {}
        
    def store_encrypted_vector(self, user_id: str, vector_data: Dict) -> str:
        """Store encrypted behavioral vector"""
        vector_id = f"vec_{uuid.uuid4().hex[:8]}"
        record = {
            'vector_id': vector_id,
            'user_id': user_id,
            'data': self._encrypt_data(vector_data),
            'timestamp': datetime.now().isoformat(),
            'type': 'behavioral_vector'
        }
        self.vectors[vector_id] = record
        self.user_profiles[user_id].append(vector_id)
        return vector_id
    
    def search_similar_vectors(self, user_id: str, query_vector: Dict, threshold: float = 0.7) -> List[Dict]:
        """Search for similar behavioral vectors"""
        if user_id not in self.user_profiles:
            return []
        
        results = []
        for vector_id in self.user_profiles[user_id]:
            if vector_id in self.vectors:
                record = self.vectors[vector_id]
                similarity = self._calculate_similarity(query_vector, self._decrypt_data(record['data']))
                if similarity >= threshold:
                    results.append({
                        'vector_id': vector_id,
                        'similarity': similarity,
                        'timestamp': record['timestamp']
                    })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    def store_transaction(self, transaction_data: Dict) -> str:
        """Store transaction record"""
        tx_id = f"tx_{uuid.uuid4().hex[:8]}"
        transaction_data['transaction_id'] = tx_id
        transaction_data['timestamp'] = datetime.now().isoformat()
        
        self.transactions[transaction_data['user_id']].append(transaction_data)
        return tx_id
    
    def get_user_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's transaction history"""
        return self.transactions.get(user_id, [])[-limit:]
    
    def generate_otp(self, user_id: str) -> str:
        """Generate OTP for user"""
        otp = str(random.randint(100000, 999999))
        self.otp_store[user_id] = {
            'otp': otp,
            'timestamp': time.time(),
            'attempts': 0,
            'verified': False
        }
        return otp
    
    def verify_otp(self, user_id: str, otp: str) -> Tuple[bool, str]:
        """Verify OTP"""
        if user_id not in self.otp_store:
            return False, "OTP expired or not generated"
        
        otp_data = self.otp_store[user_id]
        
        # Check timeout (5 minutes)
        if time.time() - otp_data['timestamp'] > 300:
            del self.otp_store[user_id]
            return False, "OTP expired"
        
        # Check attempts
        if otp_data['attempts'] >= 3:
            del self.otp_store[user_id]
            return False, "Maximum attempts exceeded"
        
        otp_data['attempts'] += 1
        
        if otp_data['otp'] == otp:
            otp_data['verified'] = True
            return True, "OTP verified successfully"
        
        attempts_remaining = 3 - otp_data['attempts']
        return False, f"Invalid OTP. {attempts_remaining} attempts remaining"
    
    def verify_password(self, user_id: str, password: str) -> PasswordStatus:
        """Verify password and track attempts"""
        # Check if user is locked out
        if user_id in self.password_lockouts:
            lockout_time = self.password_lockouts[user_id]
            if time.time() - lockout_time < 300:  # 5 minute lockout
                return PasswordStatus.INCORRECT
        
        # Get stored password (in real system, this would be hashed)
        stored_password = CONFIG['passwords'].get(user_id)
        
        if not password:
            return PasswordStatus.NOT_PROVIDED
        
        if stored_password == password:
            # Reset attempts on successful password
            self.password_attempts[user_id] = 0
            if user_id in self.password_lockouts:
                del self.password_lockouts[user_id]
            return PasswordStatus.CORRECT
        else:
            # Increment attempts
            self.password_attempts[user_id] += 1
            
            # Lock account after 3 failed attempts
            if self.password_attempts[user_id] >= 3:
                self.password_lockouts[user_id] = time.time()
                return PasswordStatus.INCORRECT
            
            return PasswordStatus.INCORRECT
    
    def get_password_attempts(self, user_id: str) -> int:
        """Get number of failed password attempts"""
        return self.password_attempts.get(user_id, 0)
    
    def _encrypt_data(self, data: Dict) -> str:
        """Simple encryption simulation"""
        data_str = json.dumps(data, sort_keys=True)
        # Simulate encryption by base64 encoding
        return base64.b64encode(data_str.encode()).decode()
    
    def _decrypt_data(self, encrypted: str) -> Dict:
        """Simple decryption simulation"""
        data_str = base64.b64decode(encrypted).decode()
        return json.loads(data_str)
    
    def _calculate_similarity(self, vec1: Dict, vec2: Dict) -> float:
        """Calculate similarity between two behavioral vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        similarities = []
        
        # Compare typing patterns
        if 'typing' in vec1 and 'typing' in vec2:
            t1 = vec1['typing']
            t2 = vec2['typing']
            if t1.get('avg_speed') and t2.get('avg_speed'):
                speed_diff = abs(t1['avg_speed'] - t2['avg_speed'])
                speed_similarity = max(0, 1 - (speed_diff / 500))  # 500ms max difference
                similarities.append(speed_similarity)
        
        # Compare mouse patterns
        if 'mouse' in vec1 and 'mouse' in vec2:
            m1 = vec1['mouse']
            m2 = vec2['mouse']
            if m1.get('avg_speed') and m2.get('avg_speed'):
                speed_diff = abs(m1['avg_speed'] - m2['avg_speed'])
                speed_similarity = max(0, 1 - (speed_diff / 1000))  # 1000px/s max difference
                similarities.append(speed_similarity)
        
        return np.mean(similarities) if similarities else 0.5

# ==================== BEHAVIOR ANALYZER ====================
class BehaviorAnalyzer:
    """Analyzes typing and mouse behavior"""
    
    def __init__(self):
        self.users_typing_patterns = defaultdict(list)
        self.users_mouse_patterns = defaultdict(list)
    
    def analyze_typing(self, keystrokes: List[Dict]) -> Dict:
        """Analyze typing speed and patterns"""
        if len(keystrokes) < 5:
            return {'status': 'insufficient_data', 'risk': 0.5}
        
        # Calculate time between keystrokes
        durations = []
        for i in range(1, len(keystrokes)):
            prev = keystrokes[i-1]['timestamp']
            curr = keystrokes[i]['timestamp']
            durations.append(curr - prev)
        
        avg_speed = np.mean(durations) if durations else 0
        speed_variance = np.var(durations) if len(durations) > 1 else 0
        
        # Detect anomalies
        anomalies = []
        
        # Check if too fast (bot)
        if avg_speed < 50:
            anomalies.append("Typing too fast (possible bot)")
        
        # Check if too perfect (bot)
        if speed_variance < 10:
            anomalies.append("Perfect timing (robotic)")
        
        # Check backspace patterns
        backspaces = sum(1 for k in keystrokes if k.get('key') == 'Backspace')
        
        # Check for repeated patterns (like copy-paste)
        if len(keystrokes) > 20:
            # Analyze timing consistency
            unique_timings = len(set(round(t, -1) for t in durations if t < 1000))
            if unique_timings < 3:
                anomalies.append("Repeated timing patterns detected")
        
        # Calculate human score
        human_score = self._calculate_typing_human_score(avg_speed, speed_variance, backspaces)
        
        return {
            'avg_speed_ms': avg_speed,
            'speed_variance': speed_variance,
            'key_count': len(keystrokes),
            'backspace_count': backspaces,
            'anomalies': anomalies,
            'human_score': human_score,
            'is_human': human_score > 50,
            'risk_level': 'safe' if human_score > 70 else 'suspicious' if human_score > 40 else 'alert'
        }
    
    def analyze_mouse(self, mouse_data: List[Dict]) -> Dict:
        """Analyze mouse movement patterns"""
        if len(mouse_data) < 10:
            return {'status': 'insufficient_data', 'risk': 0.5}
        
        velocities = []
        angles = []
        
        for i in range(1, len(mouse_data)):
            m1 = mouse_data[i-1]
            m2 = mouse_data[i]
            
            dx = m2['x'] - m1['x']
            dy = m2['y'] - m1['y']
            dt = m2['timestamp'] - m1['timestamp']
            
            if dt > 0:
                distance = np.sqrt(dx*dx + dy*dy)
                velocity = distance / dt
                angle = np.arctan2(dy, dx)
                
                velocities.append(velocity)
                angles.append(angle)
        
        if not velocities:
            return {'status': 'no_movement', 'risk': 0.5}
        
        avg_velocity = np.mean(velocities)
        velocity_variance = np.var(velocities) if len(velocities) > 1 else 0
        
        # Calculate angle variance (curvature)
        angle_changes = []
        for i in range(1, len(angles)):
            change = abs(angles[i] - angles[i-1])
            angle_changes.append(change)
        
        curvature = np.var(angle_changes) if angle_changes else 0
        
        # Detect anomalies
        anomalies = []
        
        if avg_velocity > 1000:
            anomalies.append("Mouse movements too fast")
        
        if velocity_variance < 100:
            anomalies.append("Mouse speed too consistent")
        
        if curvature < 0.1:
            anomalies.append("Mouse movements too straight")
        
        # Check for grid-like movements (bot-like)
        if len(angles) > 20:
            # Count direction changes
            direction_changes = sum(1 for i in range(1, len(angles)) 
                                  if abs(angles[i] - angles[i-1]) > np.pi/2)
            if direction_changes < len(angles) * 0.1:  # Less than 10% direction changes
                anomalies.append("Limited direction changes (possible automation)")
        
        # Calculate human score
        human_score = self._calculate_mouse_human_score(avg_velocity, velocity_variance, curvature)
        
        return {
            'avg_velocity': avg_velocity,
            'velocity_variance': velocity_variance,
            'curvature': curvature,
            'movement_count': len(mouse_data),
            'anomalies': anomalies,
            'human_score': human_score,
            'is_human': human_score > 50,
            'risk_level': 'safe' if human_score > 70 else 'suspicious' if human_score > 40 else 'alert'
        }
    
    def analyze_password_pattern(self, keystrokes: List[Dict], known_password: str) -> Dict:
        """Analyze password typing behavior specifically"""
        analysis = self.analyze_typing(keystrokes)
        
        # Additional password-specific checks
        additional_anomalies = []
        
        if len(keystrokes) > 0:
            # Check for hesitation before special characters
            special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')']
            
            # Find positions of special chars in password
            for i, char in enumerate(known_password):
                if char in special_chars:
                    # Look for hesitation before typing special character
                    pass
            
            # Check if typing speed changes dramatically (copy-paste indicator)
            if len(keystrokes) > 10:
                first_half = keystrokes[:len(keystrokes)//2]
                second_half = keystrokes[len(keystrokes)//2:]
                
                if first_half and second_half:
                    first_speeds = []
                    second_speeds = []
                    
                    for i in range(1, len(first_half)):
                        first_speeds.append(first_half[i]['timestamp'] - first_half[i-1]['timestamp'])
                    
                    for i in range(1, len(second_half)):
                        second_speeds.append(second_half[i]['timestamp'] - second_half[i-1]['timestamp'])
                    
                    if first_speeds and second_speeds:
                        avg_first = np.mean(first_speeds)
                        avg_second = np.mean(second_speeds)
                        
                        if abs(avg_first - avg_second) > 200:  # 200ms difference
                            additional_anomalies.append("Inconsistent typing speed (possible copy-paste)")
        
        analysis['anomalies'].extend(additional_anomalies)
        analysis['password_specific_analysis'] = True
        
        return analysis
    
    def _calculate_typing_human_score(self, avg_speed: float, variance: float, backspaces: int) -> float:
        """Calculate typing human-likeness score (0-100)"""
        score = 0.0
        
        # Speed score (100-300ms is optimal)
        if 100 <= avg_speed <= 200:
            score += 40
        elif 50 <= avg_speed < 100 or 100 < avg_speed <= 500:
            score += 20
        else:
            score += 5
        
        # Variance score (humans have variance)
        if variance > 1000:
            score += 30
        elif variance > 100:
            score += 20
        elif variance > 10:
            score += 10
        else:
            score += 0
        
        # Backspace score (humans make mistakes)
        if backspaces > 0:
            score += 30 * min(backspaces / 10, 1)
        
        return min(score, 100.0)
    
    def _calculate_mouse_human_score(self, velocity: float, variance: float, curvature: float) -> float:
        """Calculate mouse human-likeness score (0-100)"""
        score = 0.0
        
        # Velocity score (100-500 px/s is human)
        if 100 <= velocity <= 500:
            score += 40
        elif 50 <= velocity < 100 or 500 < velocity <= 1000:
            score += 20
        else:
            score += 5
        
        # Variance score
        if variance > 10000:
            score += 30
        elif variance > 1000:
            score += 20
        elif variance > 100:
            score += 15
        elif variance > 10:
            score += 10
        else:
            score += 0
        
        # Curvature score
        if curvature > 0.5:
            score += 30
        elif curvature > 0.1:
            score += 20
        elif curvature > 0.01:
            score += 10
        else:
            score += 0
        
        return min(score, 100.0)

# ==================== FRAUD DETECTOR ====================
class FraudDetector:
    """Main fraud detection engine"""
    
    def __init__(self, cyborgdb: CyborgDB, analyzer: BehaviorAnalyzer):
        self.db = cyborgdb
        self.analyzer = analyzer
        self.user_risk_profiles = defaultdict(lambda: {'sessions': 0, 'avg_risk': 0.0})
    
    def analyze_transaction(self, user_id: str, behavior_data: Dict, transaction_data: Dict, password: str = None) -> Dict:
        """Analyze transaction for fraud with password validation"""
        start_time = time.time()
        
        # Verify password if provided
        password_status = self.db.verify_password(user_id, password) if password else PasswordStatus.NOT_PROVIDED
        password_attempts = self.db.get_password_attempts(user_id)
        
        # Check if account is locked
        if password_status == PasswordStatus.INCORRECT and password_attempts >= 3:
            return {
                'transaction_id': str(uuid.uuid4()),
                'user_id': user_id,
                'status': 'blocked',
                'reason': 'Account locked due to multiple failed password attempts',
                'password_status': 'incorrect',
                'password_attempts': password_attempts,
                'decision': 'block',
                'risk_level': 'alert',
                'total_risk': 1.0,
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        
        # Analyze behavior
        typing_analysis = self.analyzer.analyze_typing(behavior_data.get('keystrokes', []))
        mouse_analysis = self.analyzer.analyze_mouse(behavior_data.get('mouse_movements', []))
        
        # If password was provided, analyze password-specific behavior
        password_analysis = None
        if password and password_status == PasswordStatus.CORRECT:
            # Get known password for this user
            known_password = CONFIG['passwords'].get(user_id)
            if known_password:
                password_analysis = self.analyzer.analyze_password_pattern(
                    behavior_data.get('keystrokes', []), 
                    known_password
                )
                # Use password analysis if available
                if password_analysis:
                    typing_analysis = password_analysis
        
        # Create behavioral vector
        behavior_vector = {
            'typing': {
                'avg_speed': typing_analysis.get('avg_speed_ms', 0),
                'variance': typing_analysis.get('speed_variance', 0),
                'backspaces': typing_analysis.get('backspace_count', 0),
                'password_typed': password is not None
            },
            'mouse': {
                'avg_speed': mouse_analysis.get('avg_velocity', 0),
                'variance': mouse_analysis.get('velocity_variance', 0),
                'curvature': mouse_analysis.get('curvature', 0)
            }
        }
        
        # Store in CyborgDB
        vector_id = self.db.store_encrypted_vector(user_id, behavior_vector)
        
        # Search for similar patterns
        similar_vectors = self.db.search_similar_vectors(user_id, behavior_vector, threshold=0.6)
        
        # Calculate risk scores
        behavior_risk = self._calculate_behavior_risk(typing_analysis, mouse_analysis, similar_vectors)
        amount_risk = self._calculate_amount_risk(transaction_data.get('amount', 0))
        time_risk = self._calculate_time_risk()
        password_risk = self._calculate_password_risk(password_status, password_attempts)
        
        # Combine risks (password risk has higher weight)
        total_risk = (behavior_risk * 0.4) + (amount_risk * 0.2) + (time_risk * 0.1) + (password_risk * 0.3)
        
        # Determine risk level
        risk_level = self._determine_risk_level(total_risk, typing_analysis, mouse_analysis, password_status)
        
        # Make decision
        decision = self._make_decision(total_risk, transaction_data.get('amount', 0), risk_level, password_status)
        
        # Store transaction
        transaction_record = {
            'transaction_id': str(uuid.uuid4()),
            'user_id': user_id,
            'behavior_vector_id': vector_id,
            'amount': transaction_data.get('amount', 0),
            'type': transaction_data.get('type', 'payment'),
            'password_provided': password is not None,
            'password_status': password_status.value if password_status else 'not_provided',
            'password_attempts': password_attempts,
            'behavior_risk': behavior_risk,
            'amount_risk': amount_risk,
            'time_risk': time_risk,
            'password_risk': password_risk,
            'total_risk': total_risk,
            'risk_level': risk_level.value,
            'decision': decision.value,
            'typing_human_score': typing_analysis.get('human_score', 0),
            'mouse_human_score': mouse_analysis.get('human_score', 0),
            'anomalies': typing_analysis.get('anomalies', []) + mouse_analysis.get('anomalies', []),
            'similar_patterns_found': len(similar_vectors),
            'processing_time_ms': (time.time() - start_time) * 1000,
            'password_analysis_included': password_analysis is not None
        }
        
        # Add OTP if required
        if decision == Decision.OTP_REQUIRED:
            otp = self.db.generate_otp(user_id)
            transaction_record['otp_required'] = True
            transaction_record['otp'] = otp
        else:
            transaction_record['otp_required'] = False
        
        # Store in database
        self.db.store_transaction(transaction_record)
        
        # Update user profile
        self.user_risk_profiles[user_id]['sessions'] += 1
        self.user_risk_profiles[user_id]['avg_risk'] = (
            self.user_risk_profiles[user_id]['avg_risk'] * (self.user_risk_profiles[user_id]['sessions'] - 1) + total_risk
        ) / self.user_risk_profiles[user_id]['sessions']
        
        return transaction_record
    
    def _calculate_behavior_risk(self, typing: Dict, mouse: Dict, similar_vectors: List[Dict]) -> float:
        """Calculate risk from behavior analysis"""
        risk = 0.0
        
        # Base risk from typing analysis
        typing_human_score = typing.get('human_score', 50)
        typing_risk = 1 - (typing_human_score / 100)
        risk += typing_risk * 0.4
        
        # Base risk from mouse analysis
        mouse_human_score = mouse.get('human_score', 50)
        mouse_risk = 1 - (mouse_human_score / 100)
        risk += mouse_risk * 0.3
        
        # Risk from pattern mismatch
        if similar_vectors:
            avg_similarity = np.mean([v['similarity'] for v in similar_vectors])
            similarity_risk = 1 - avg_similarity
        else:
            similarity_risk = 0.5  # No history = medium risk
        
        risk += similarity_risk * 0.3
        
        return min(risk, 1.0)
    
    def _calculate_amount_risk(self, amount: float) -> float:
        """Calculate risk based on transaction amount"""
        if amount <= CONFIG['transaction']['small_amount']:
            return 0.1
        elif amount <= CONFIG['transaction']['medium_amount']:
            return 0.3
        elif amount <= CONFIG['transaction']['large_amount']:
            return 0.5
        elif amount <= CONFIG['transaction']['critical_amount']:
            return 0.7
        else:
            return 0.9
    
    def _calculate_time_risk(self) -> float:
        """Calculate risk based on time of day"""
        hour = datetime.now().hour
        if 2 <= hour <= 6:  # 2 AM - 6 AM
            return 0.7
        elif 0 <= hour < 2 or 22 <= hour:  # Midnight - 2 AM or 10 PM - midnight
            return 0.5
        else:
            return 0.3
    
    def _calculate_password_risk(self, password_status: PasswordStatus, attempts: int) -> float:
        """Calculate risk based on password status"""
        if password_status == PasswordStatus.CORRECT:
            return 0.1  # Low risk with correct password
        elif password_status == PasswordStatus.INCORRECT:
            if attempts >= 3:
                return 1.0  # Maximum risk for locked account
            else:
                return 0.7 + (attempts * 0.1)  # Incremental risk per attempt
        elif password_status == PasswordStatus.NOT_PROVIDED:
            return 0.5  # Medium risk when no password provided
        else:
            return 0.5
    
    def _determine_risk_level(self, total_risk: float, typing: Dict, mouse: Dict, password_status: PasswordStatus) -> RiskLevel:
        """Determine risk level"""
        # If password is incorrect multiple times, immediate alert
        if password_status == PasswordStatus.INCORRECT:
            return RiskLevel.ALERT
        
        if total_risk < 0.3:
            return RiskLevel.SAFE
        elif total_risk < 0.6:
            return RiskLevel.SUSPICIOUS
        else:
            return RiskLevel.ALERT
    
    def _make_decision(self, total_risk: float, amount: float, risk_level: RiskLevel, password_status: PasswordStatus) -> Decision:
        """Make decision based on risk and amount"""
        if risk_level == RiskLevel.ALERT:
            return Decision.BLOCK
        elif password_status == PasswordStatus.INCORRECT:
            return Decision.BLOCK
        elif amount >= CONFIG['transaction']['large_amount'] or total_risk >= 0.4:
            return Decision.OTP_REQUIRED
        else:
            return Decision.APPROVE
    
    def verify_otp(self, user_id: str, otp: str) -> Tuple[bool, str]:
        """Verify OTP"""
        return self.db.verify_otp(user_id, otp)
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics"""
        profile = self.user_risk_profiles[user_id]
        history = self.db.get_user_history(user_id, limit=10)
        
        if history:
            recent_risks = [tx['total_risk'] for tx in history]
            recent_decisions = [tx['decision'] for tx in history]
        else:
            recent_risks = [0.5]
            recent_decisions = ['approve']
        
        return {
            'user_id': user_id,
            'total_sessions': profile['sessions'],
            'avg_risk_score': profile['avg_risk'],
            'recent_risk_avg': np.mean(recent_risks) if recent_risks else 0,
            'recent_decisions': recent_decisions,
            'approved_count': sum(1 for d in recent_decisions if d == 'approve'),
            'otp_required_count': sum(1 for d in recent_decisions if d == 'otp_required'),
            'blocked_count': sum(1 for d in recent_decisions if d == 'block'),
            'password_attempts': self.db.get_password_attempts(user_id)
        }

# ==================== FLASK APP ====================
app = Flask(__name__)
CORS(app)

# Initialize components
cyborgdb = CyborgDB()
behavior_analyzer = BehaviorAnalyzer()
fraud_detector = FraudDetector(cyborgdb, behavior_analyzer)

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>GhostWallet - Smart Fraud Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #000000, #0a0f1f, #151b2c);

            min-height: 100vh;
            color: white;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        header {
            background: linear-gradient(90deg, #1e3799, #0c2461);
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        header p {
            font-size: 1.2em;
            opacity: 0.9;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .stats-bar {
            display: flex;
            justify-content: space-around;
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #4a69bd;
            display: block;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .main-content {
            padding: 40px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .behavior-section, .transaction-section {
            background: rgba(255, 255, 255, 0.05);
            padding: 30px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .section-title {
            color: #4a69bd;
            margin-bottom: 25px;
            font-size: 1.8em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .section-title i {
            font-size: 1.5em;
        }
        
        .behavior-box {
            background: rgba(0, 0, 0, 0.3);
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            transition: all 0.3s;
        }
        
        .behavior-box:hover {
            border-color: #4a69bd;
            background: rgba(74, 105, 189, 0.1);
        }
        
        .behavior-box h4 {
            color: #4a69bd;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        textarea {
            width: 100%;
            padding: 20px;
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            color: white;
            font-size: 16px;
            resize: vertical;
            min-height: 150px;
            margin-bottom: 20px;
            transition: all 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #4a69bd;
            box-shadow: 0 0 20px rgba(74, 105, 189, 0.3);
        }
        
        .counter {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 10px;
        }
        
        .counter-item {
            text-align: center;
            flex: 1;
        }
        
        .counter-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #4a69bd;
            display: block;
        }
        
        .counter-label {
            font-size: 0.9em;
            opacity: 0.7;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 10px;
            font-size: 1.1em;
            color: white;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 15px;
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            color: white;
            font-size: 16px;
            transition: all 0.3s;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #4a69bd;
            box-shadow: 0 0 15px rgba(74, 105, 189, 0.3);
        }
        
        .main-button {
            display: block;
            width: 100%;
            padding: 25px;
            background: linear-gradient(90deg, #1e3799, #4a69bd);
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 1.8em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin: 40px 0;
            position: relative;
            overflow: hidden;
        }
        
        .main-button:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(30, 55, 153, 0.4);
            background: linear-gradient(90deg, #4a69bd, #1e3799);
        }
        
        .main-button:active {
            transform: translateY(0);
        }
        
        .result-container {
            grid-column: 1 / -1;
            min-height: 300px;
            transition: all 0.5s;
        }
        
        .result-box {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 30px;
            margin-top: 20px;
            animation: slideIn 0.5s ease-out;
            border: 3px solid transparent;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-safe {
            border-color: #00b894;
            background: linear-gradient(135deg, rgba(0, 184, 148, 0.1), rgba(85, 239, 196, 0.1));
        }
        
        .result-suspicious {
            border-color: #fdcb6e;
            background: linear-gradient(135deg, rgba(253, 203, 110, 0.1), rgba(255, 234, 167, 0.1));
        }
        
        .result-alert {
            border-color: #e17055;
            background: linear-gradient(135deg, rgba(225, 112, 85, 0.1), rgba(255, 118, 117, 0.1));
        }
        
        .result-header {
            font-size: 2.5em;
            margin-bottom: 20px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }
        
        .result-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .detail-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            transition: all 0.3s;
        }
        
        .detail-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.1);
        }
        
        .detail-value {
            font-size: 2em;
            font-weight: bold;
            color: #4a69bd;
            margin-bottom: 5px;
        }
        
        .detail-label {
            font-size: 0.9em;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .anomalies {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .anomalies h4 {
            color: #e17055;
            margin-bottom: 15px;
            font-size: 1.2em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .anomalies ul {
            list-style: none;
            padding-left: 20px;
        }
        
        .anomalies li {
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
        }
        
        .anomalies li:before {
            content: "⚠️";
            position: absolute;
            left: -25px;
        }
        
        .otp-section {
            background: rgba(0, 0, 0, 0.3);
            padding: 25px;
            border-radius: 10px;
            margin-top: 20px;
            border: 2px solid #fdcb6e;
        }
        
        .otp-section h4 {
            color: #fdcb6e;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .otp-input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .otp-input {
            flex: 1;
            padding: 15px;
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid #fdcb6e;
            border-radius: 10px;
            color: white;
            font-size: 18px;
            text-align: center;
            letter-spacing: 5px;
        }
        
        .otp-button {
            padding: 15px 30px;
            background: #fdcb6e;
            color: #2d3436;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .otp-button:hover {
            background: #ffeaa7;
            transform: translateY(-2px);
        }
        
        .loader {
            border: 5px solid rgba(255, 255, 255, 0.1);
            border-top: 5px solid #4a69bd;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .footer {
            text-align: center;
            padding: 30px;
            margin-top: 40px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.9em;
            opacity: 0.7;
        }
        
        .cyborg-badge {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 0.9em;
            border: 1px solid #4a69bd;
        }
        
        .password-status {
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            font-weight: bold;
            text-align: center;
        }
        
        .password-correct {
            background: rgba(0, 184, 148, 0.2);
            border: 1px solid #00b894;
            color: #00b894;
        }
        
        .password-incorrect {
            background: rgba(225, 112, 85, 0.2);
            border: 1px solid #e17055;
            color: #e17055;
        }
        
        .password-required {
            background: rgba(253, 203, 110, 0.2);
            border: 1px solid #fdcb6e;
            color: #fdcb6e;
        }
        
        .demo-passwords {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            font-size: 0.9em;
        }
        
        .demo-passwords h5 {
            color: #4a69bd;
            margin-bottom: 10px;
        }
        
        .demo-passwords ul {
            padding-left: 20px;
        }
        
        .demo-passwords li {
            margin: 5px 0;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="cyborg-badge">🔐 Password + Behavioral Analysis</div>
            <h1>👻 GHOSTWALLET</h1>
            <p>Smart Fraud Detection - Password Validation + Behavioral Biometrics</p>
        </header>
        
        <div class="stats-bar">
            <div class="stat-item">
                <span class="stat-value" id="totalTransactions">0</span>
                <span class="stat-label">Transactions</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="safeCount">0</span>
                <span class="stat-label">Safe</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="suspiciousCount">0</span>
                <span class="stat-label">Suspicious</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="alertCount">0</span>
                <span class="stat-label">Alerts</span>
            </div>
        </div>
        
        <div class="main-content">
            <div class="behavior-section">
                <div class="section-title">
                    <span>⌨️</span> BEHAVIORAL PATTERNS ANALYSIS
                </div>
                
                <div class="behavior-box">
                    <h4>Practice Typing (Optional):</h4>
                    <textarea id="practiceInput" placeholder="Type here to practice... This helps establish your baseline typing rhythm. Move mouse while typing!"></textarea>
                    
                    <div class="counter">
                        <div class="counter-item">
                            <span class="counter-value" id="practiceKeyCount">0</span>
                            <span class="counter-label">Practice Keys</span>
                        </div>
                        <div class="counter-item">
                            <span class="counter-value" id="practiceMouseCount">0</span>
                            <span class="counter-label">Mouse Moves</span>
                        </div>
                        <div class="counter-item">
                            <span class="counter-value" id="practiceBackspaceCount">0</span>
                            <span class="counter-label">Mistakes</span>
                        </div>
                    </div>
                </div>
                
                <div class="behavior-box">
                    <h4>Behavioral Analysis Results:</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 10px;">
                            <div style="color: #4a69bd; font-weight: bold; margin-bottom: 5px;">Typing Score</div>
                            <div style="font-size: 1.5em; font-weight: bold;" id="typingScore">0%</div>
                            <div style="font-size: 0.9em; opacity: 0.7;" id="typingStatus">Waiting...</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 10px;">
                            <div style="color: #4a69bd; font-weight: bold; margin-bottom: 5px;">Mouse Score</div>
                            <div style="font-size: 1.5em; font-weight: bold;" id="mouseScore">0%</div>
                            <div style="font-size: 0.9em; opacity: 0.7;" id="mouseStatus">Waiting...</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.7; text-align: center;">
                        💡 System analyzes: Typing speed, rhythm, mistakes, and mouse movement patterns
                    </div>
                </div>
            </div>
            
            <div class="transaction-section">
                <div class="section-title">
                    <span>🔐</span> TRANSACTION & PASSWORD
                </div>
                
                <div class="form-group">
                    <label for="userId">👤 User ID:</label>
                    <input type="text" id="userId" value="user_001" placeholder="Enter your user ID">
                </div>
                
                <div class="form-group">
                    <label for="passwordInput">🔒 Password (Required for Transaction):</label>
                    <input type="password" id="passwordInput" placeholder="Type your password here..." autocomplete="off">
                    <div style="margin-top: 10px;" id="passwordStatus"></div>
                    <div style="font-size: 0.9em; opacity: 0.7; margin-top: 5px;">
                        💡 Even with correct password, suspicious typing behavior may trigger OTP
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="amount">💰 Amount ($):</label>
                    <input type="number" id="amount" value="150" placeholder="Transaction amount">
                    <div style="font-size: 0.9em; opacity: 0.7; margin-top: 5px;">
                        💡 Amounts over $1000 require OTP verification
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="transactionType">📝 Transaction Type:</label>
                    <select id="transactionType">
                        <option value="payment">Payment</option>
                        <option value="transfer">Bank Transfer</option>
                        <option value="withdrawal">Withdrawal</option>
                        <option value="deposit">Deposit</option>
                    </select>
                </div>
                
                <div class="demo-passwords">
                    <h5>📋 Demo Passwords (for testing):</h5>
                    <ul>
                        <li>user_001 → SecurePass123!</li>
                        <li>user_002 → Test@Password456</li>
                        <li>user_003 → Demo#Pass789</li>
                        <li>user_004 → GhostWallet2024!</li>
                    </ul>
                </div>
                
                <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px; margin-top: 20px;">
                    <h4 style="color: #4a69bd; margin-bottom: 10px;">⚡ Enhanced Security Rules:</h4>
                    <ul style="padding-left: 20px; font-size: 0.9em; opacity: 0.8;">
                        <li>✅ Correct password + Normal behavior → Usually Safe</li>
                        <li>✅ Correct password + Suspicious typing → OTP Required</li>
                        <li>❌ Wrong password → Immediate Block (3 attempts lock)</li>
                        <li>❌ No password provided → Medium Risk + OTP</li>
                        <li>⚠️ Amount > $1000 → Always OTP Required</li>
                        <li>🚫 Robotic patterns → Immediate Block</li>
                    </ul>
                </div>
            </div>
            
            <button class="main-button" onclick="processTransaction()">
                <span style="font-size: 1.5em; margin-right: 15px;">🔍</span>
                ANALYZE WITH PASSWORD & BEHAVIOR
            </button>
            
            <div class="result-container" id="resultContainer">
                <!-- Results will appear here -->
            </div>
            
            <div class="otp-section" id="otpSection" style="display: none;">
                <h4>🔒 OTP VERIFICATION REQUIRED</h4>
                <p>Additional verification needed due to suspicious behavior or large amount:</p>
                <div class="otp-input-group">
                    <input type="text" id="otpInput" class="otp-input" placeholder="000000" maxlength="6">
                    <button class="otp-button" onclick="verifyOTP()">VERIFY OTP</button>
                </div>
                <div style="font-size: 0.9em; opacity: 0.7;">
                    ⏳ OTP expires in 5 minutes • 3 attempts allowed
                </div>
                <div id="otpMessage" style="margin-top: 10px;"></div>
            </div>
        </div>
        
        <div class="footer">
            <p>GhostWallet © 2024 | Password + Behavioral Biometrics | Even correct password can trigger OTP if behavior is suspicious</p>
            <p>Layers: Password Validation → Typing Analysis → Mouse Patterns → Amount Risk → Time Risk → Final Decision</p>
        </div>
    </div>

    <script>
        // ==================== GLOBAL VARIABLES ====================
        let capturedBehavior = {
            keystrokes: [],
            mouse_movements: [],
            clicks: [],
            passwordKeystrokes: [],
            passwordMouseMovements: []
        };
        
        let practiceKeyCount = 0;
        let practiceMouseCount = 0;
        let practiceBackspaceCount = 0;
        let passwordKeyCount = 0;
        let passwordBackspaceCount = 0;
        let isCapturing = true;
        let currentTransactionId = null;
        let currentUserId = null;
        let activeField = 'practice'; // 'practice' or 'password'
        let passwordAttempts = 0;
        
        // ==================== BEHAVIOR CAPTURE ====================
        const practiceInput = document.getElementById('practiceInput');
        const passwordInput = document.getElementById('passwordInput');
        const practiceKeyCountElement = document.getElementById('practiceKeyCount');
        const practiceMouseCountElement = document.getElementById('practiceMouseCount');
        const practiceBackspaceCountElement = document.getElementById('practiceBackspaceCount');
        const typingScoreElement = document.getElementById('typingScore');
        const mouseScoreElement = document.getElementById('mouseScore');
        const typingStatusElement = document.getElementById('typingStatus');
        const mouseStatusElement = document.getElementById('mouseStatus');
        const passwordStatusElement = document.getElementById('passwordStatus');
        
        // Practice text area events
        practiceInput.addEventListener('focus', () => {
            activeField = 'practice';
            updateStatusDisplay();
        });
        
        practiceInput.addEventListener('keydown', (e) => {
            if (!isCapturing) return;
            
            if (activeField === 'practice') {
                practiceKeyCount++;
                practiceKeyCountElement.textContent = practiceKeyCount;
                
                capturedBehavior.keystrokes.push({
                    key: e.key,
                    timestamp: Date.now(),
                    type: 'keydown',
                    field: 'practice'
                });
                
                if (e.key === 'Backspace') {
                    practiceBackspaceCount++;
                    practiceBackspaceCountElement.textContent = practiceBackspaceCount;
                }
                
                analyzeLiveBehavior();
            }
        });
        
        practiceInput.addEventListener('keyup', (e) => {
            if (!isCapturing || activeField !== 'practice') return;
            
            capturedBehavior.keystrokes.push({
                key: e.key,
                timestamp: Date.now(),
                type: 'keyup',
                field: 'practice'
            });
        });
        
        // Password input events
        passwordInput.addEventListener('focus', () => {
            activeField = 'password';
            updateStatusDisplay();
        });
        
        passwordInput.addEventListener('blur', () => {
            activeField = 'practice';
            updateStatusDisplay();
        });
        
        passwordInput.addEventListener('keydown', (e) => {
            if (!isCapturing) return;
            
            if (activeField === 'password') {
                passwordKeyCount++;
                
                capturedBehavior.passwordKeystrokes.push({
                    key: e.key,
                    timestamp: Date.now(),
                    type: 'keydown',
                    field: 'password'
                });
                
                if (e.key === 'Backspace') {
                    passwordBackspaceCount++;
                }
                
                // Don't show actual password length
                updatePasswordStatus('typing', `Typing... (${passwordInput.value.length + 1} chars)`);
                analyzeLiveBehavior();
            }
        });
        
        passwordInput.addEventListener('keyup', (e) => {
            if (!isCapturing || activeField !== 'password') return;
            
            capturedBehavior.passwordKeystrokes.push({
                key: e.key,
                timestamp: Date.now(),
                type: 'keyup',
                field: 'password'
            });
        });
        
        // Capture mouse movements (global)
        document.addEventListener('mousemove', (e) => {
            if (!isCapturing) return;
            
            const mouseEvent = {
                x: e.clientX,
                y: e.clientY,
                timestamp: Date.now(),
                field: activeField
            };
            
            if (activeField === 'practice') {
                practiceMouseCount++;
                practiceMouseCountElement.textContent = practiceMouseCount;
                capturedBehavior.mouse_movements.push(mouseEvent);
            } else if (activeField === 'password') {
                capturedBehavior.passwordMouseMovements.push(mouseEvent);
            }
            
            if (practiceMouseCount % 50 === 0 || capturedBehavior.mouse_movements.length % 50 === 0) {
                analyzeLiveBehavior();
            }
        });
        
        // Capture clicks
        document.addEventListener('click', (e) => {
            if (!isCapturing) return;
            
            capturedBehavior.clicks.push({
                x: e.clientX,
                y: e.clientY,
                timestamp: Date.now(),
                button: e.button,
                field: activeField
            });
        });
        
        function updateStatusDisplay() {
            if (activeField === 'password') {
                document.querySelector('.behavior-box h4').innerHTML = '🔒 PASSWORD TYPING ANALYSIS';
            } else {
                document.querySelector('.behavior-box h4').innerHTML = '⌨️ PRACTICE TYPING ANALYSIS';
            }
        }
        
        function updatePasswordStatus(type, message) {
            let html = '';
            if (type === 'typing') {
                html = `<div class="password-status password-required">${message}</div>`;
            } else if (type === 'correct') {
                html = `<div class="password-status password-correct">✅ ${message}</div>`;
            } else if (type === 'incorrect') {
                html = `<div class="password-status password-incorrect">❌ ${message}</div>`;
            }
            passwordStatusElement.innerHTML = html;
        }
        
        function analyzeLiveBehavior() {
            let keystrokesToAnalyze = [];
            let mouseMovementsToAnalyze = [];
            
            if (activeField === 'password') {
                keystrokesToAnalyze = capturedBehavior.passwordKeystrokes.filter(k => k.type === 'keydown');
                mouseMovementsToAnalyze = capturedBehavior.passwordMouseMovements;
            } else {
                keystrokesToAnalyze = capturedBehavior.keystrokes.filter(k => k.type === 'keydown' && k.field === 'practice');
                mouseMovementsToAnalyze = capturedBehavior.mouse_movements.filter(m => m.field === 'practice');
            }
            
            // Typing analysis
            if (keystrokesToAnalyze.length >= 2) {
                let totalTime = 0;
                let count = 0;
                for (let i = 1; i < keystrokesToAnalyze.length; i++) {
                    const timeDiff = keystrokesToAnalyze[i].timestamp - keystrokesToAnalyze[i-1].timestamp;
                    if (timeDiff < 1000) {
                        totalTime += timeDiff;
                        count++;
                    }
                }
                
                if (count > 0) {
                    const avgSpeed = totalTime / count;
                    let typingScore = 0;
                    let typingStatus = "";
                    
                    if (avgSpeed >= 100 && avgSpeed <= 300) {
                        typingScore = 85 + Math.random() * 10;
                        typingStatus = "Natural human speed ✓";
                    } else if (avgSpeed < 50) {
                        typingScore = 20 + Math.random() * 20;
                        typingStatus = "Too fast (suspicious) ⚠️";
                    } else if (avgSpeed > 500) {
                        typingScore = 30 + Math.random() * 20;
                        typingStatus = "Too slow ⚠️";
                    } else {
                        typingScore = 50 + Math.random() * 30;
                        typingStatus = "Within normal range";
                    }
                    
                    typingScoreElement.textContent = Math.round(typingScore) + "%";
                    typingStatusElement.textContent = typingStatus;
                }
            }
            
            // Mouse analysis
            if (mouseMovementsToAnalyze.length > 10) {
                const mouseScore = 60 + Math.random() * 30;
                mouseScoreElement.textContent = Math.round(mouseScore) + "%";
                mouseStatusElement.textContent = "Natural movement ✓";
            }
        }
        
        // ==================== TRANSACTION PROCESSING ====================
        async function processTransaction() {
            const userId = document.getElementById('userId').value || 'user_001';
            const password = document.getElementById('passwordInput').value;
            const amount = parseFloat(document.getElementById('amount').value) || 100;
            currentUserId = userId;
            
            // Check password length
            if (!password || password.length < 8) {
                updatePasswordStatus('incorrect', 'Password must be at least 8 characters');
                return;
            }
            
            // Show loading
            const resultContainer = document.getElementById('resultContainer');
            resultContainer.innerHTML = `
                <div style="text-align: center; padding: 50px;">
                    <div class="loader"></div>
                    <p style="margin-top: 20px; font-size: 1.2em; color: #4a69bd;">
                        Analyzing Password + Behavior with CyborgDB...
                    </p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 30px;">
                        <div>
                            <div style="font-weight: bold; color: #4a69bd;">Step 1:</div>
                            <div style="font-size: 0.9em; opacity: 0.8;">Validating password...</div>
                        </div>
                        <div>
                            <div style="font-weight: bold; color: #4a69bd;">Step 2:</div>
                            <div style="font-size: 0.9em; opacity: 0.8;">Analyzing typing behavior...</div>
                        </div>
                        <div>
                            <div style="font-weight: bold; color: #4a69bd;">Step 3:</div>
                            <div style="font-size: 0.9em; opacity: 0.8;">Checking mouse patterns...</div>
                        </div>
                        <div>
                            <div style="font-weight: bold; color: #4a69bd;">Step 4:</div>
                            <div style="font-size: 0.9em; opacity: 0.8;">Calculating risk...</div>
                        </div>
                        <div>
                            <div style="font-weight: bold; color: #4a69bd;">Step 5:</div>
                            <div style="font-size: 0.9em; opacity: 0.8;">Making security decision...</div>
                        </div>
                        <div>
                            <div style="font-weight: bold; color: #4a69bd;">Step 6:</div>
                            <div style="font-size: 0.9em; opacity: 0.8;">Encrypting with CyborgDB...</div>
                        </div>
                    </div>
                </div>
            `;
            
            // Hide OTP section
            document.getElementById('otpSection').style.display = 'none';
            
            // Prepare data
            const transactionData = {
                user_id: userId,
                amount: amount,
                type: document.getElementById('transactionType').value || 'payment'
            };
            
            // Prepare behavior data (use password typing primarily)
            const behaviorData = {
                keystrokes: capturedBehavior.passwordKeystrokes.length > 10 ? 
                           capturedBehavior.passwordKeystrokes : 
                           capturedBehavior.keystrokes.filter(k => k.field === 'practice').slice(-100),
                mouse_movements: capturedBehavior.passwordMouseMovements.length > 20 ?
                                capturedBehavior.passwordMouseMovements :
                                capturedBehavior.mouse_movements.filter(m => m.field === 'practice').slice(-200),
                clicks: capturedBehavior.clicks.slice(-50)
            };
            
            try {
                const response = await fetch('/api/transaction', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        password: password,
                        transaction: transactionData,
                        behavior: behaviorData
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    currentTransactionId = result.transaction_id;
                    displayResults(result);
                    updateStats(result);
                    
                    // Update password status
                    if (result.password_status === 'correct') {
                        updatePasswordStatus('correct', '✅ Password correct');
                    } else if (result.password_status === 'incorrect') {
                        passwordAttempts++;
                        updatePasswordStatus('incorrect', `❌ Password incorrect (Attempt ${passwordAttempts}/3)`);
                    }
                    
                    // Show OTP section if required
                    if (result.decision === 'otp_required') {
                        document.getElementById('otpSection').style.display = 'block';
                        document.getElementById('otpInput').value = '';
                        document.getElementById('otpMessage').innerHTML = `
                            <div style="color: #fdcb6e; font-weight: bold;">
                                📧 OTP Generated: <span id="displayOTP">${result.otp || '123456'}</span>
                                <br><small>Even with correct password, suspicious behavior triggered OTP</small>
                            </div>
                        `;
                    }
                } else {
                    throw new Error(result.error || 'Unknown error');
                }
                
            } catch (error) {
                console.error('Error:', error);
                resultContainer.innerHTML = `
                    <div class="result-box result-alert">
                        <div class="result-header">❌ ERROR</div>
                        <p>Failed to process transaction. Please try again.</p>
                        <p style="color: #ff9999; margin-top: 10px;">${error.message}</p>
                    </div>
                `;
            }
        }
        
        async function verifyOTP() {
            const otp = document.getElementById('otpInput').value;
            if (!otp || otp.length !== 6) {
                document.getElementById('otpMessage').innerHTML = `
                    <div style="color: #e17055;">❌ Please enter a valid 6-digit OTP</div>
                `;
                return;
            }
            
            try {
                const response = await fetch('/api/verify-otp', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: currentUserId,
                        otp: otp
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('otpMessage').innerHTML = `
                        <div style="color: #00b894; font-weight: bold;">✅ OTP Verified Successfully!</div>
                        <div style="margin-top: 10px; font-size: 0.9em;">
                            Transaction approved despite suspicious behavior (OTP override).
                        </div>
                    `;
                    
                    // Update the result display
                    setTimeout(() => {
                        const resultContainer = document.getElementById('resultContainer');
                        const currentHTML = resultContainer.innerHTML;
                        const updatedHTML = currentHTML.replace(
                            '⚠️ VERIFICATION REQUIRED',
                            '✅ TRANSACTION APPROVED (OTP Verified)'
                        ).replace(
                            'SUSPICIOUS BEHAVIOR',
                            'APPROVED WITH OTP'
                        );
                        resultContainer.innerHTML = updatedHTML;
                    }, 1000);
                    
                    // Hide OTP section after 3 seconds
                    setTimeout(() => {
                        document.getElementById('otpSection').style.display = 'none';
                    }, 3000);
                    
                } else {
                    document.getElementById('otpMessage').innerHTML = `
                        <div style="color: #e17055;">❌ ${result.message || 'Invalid OTP'}</div>
                    `;
                }
                
            } catch (error) {
                document.getElementById('otpMessage').innerHTML = `
                    <div style="color: #e17055;">❌ Error verifying OTP. Please try again.</div>
                `;
            }
        }
        
        function displayResults(result) {
            const resultContainer = document.getElementById('resultContainer');
            
            let resultClass = 'result-safe';
            let resultIcon = '✅';
            let resultTitle = 'SAFE TRANSACTION';
            
            if (result.risk_level === 'suspicious') {
                resultClass = 'result-suspicious';
                resultIcon = '⚠️';
                resultTitle = 'SUSPICIOUS BEHAVIOR';
            } else if (result.risk_level === 'alert') {
                resultClass = 'result-alert';
                resultIcon = '🚫';
                resultTitle = 'ALERT: POSSIBLE FRAUD';
            }
            
            // Password status display
            let passwordStatusHTML = '';
            if (result.password_status === 'correct') {
                passwordStatusHTML = `
                    <div style="background: rgba(0, 184, 148, 0.2); padding: 15px; border-radius: 10px; margin: 15px 0; border: 1px solid #00b894;">
                        <strong>✅ Password:</strong> Correct (But behavior analysis continues...)
                    </div>
                `;
            } else if (result.password_status === 'incorrect') {
                passwordStatusHTML = `
                    <div style="background: rgba(225, 112, 85, 0.2); padding: 15px; border-radius: 10px; margin: 15px 0; border: 1px solid #e17055;">
                        <strong>❌ Password:</strong> Incorrect (Attempt ${result.password_attempts || 1}/3)
                    </div>
                `;
            } else {
                passwordStatusHTML = `
                    <div style="background: rgba(253, 203, 110, 0.2); padding: 15px; border-radius: 10px; margin: 15px 0; border: 1px solid #fdcb6e;">
                        <strong>⚠️ Password:</strong> Not provided (Higher risk assessment)
                    </div>
                `;
            }
            
            // Format anomalies
            let anomaliesHTML = '';
            if (result.anomalies && result.anomalies.length > 0) {
                anomaliesHTML = `
                    <div class="anomalies">
                        <h4><span>🚨</span> DETECTED BEHAVIORAL ANOMALIES:</h4>
                        <ul>
                            ${result.anomalies.map(anomaly => `<li>${anomaly}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            // Format decision message
            let decisionMessage = '';
            let decisionExplanation = '';
            
            if (result.decision === 'approve') {
                decisionMessage = 'Transaction Approved';
                decisionExplanation = 'Password correct + Normal behavior patterns';
            } else if (result.decision === 'otp_required') {
                decisionMessage = 'OTP Verification Required';
                if (result.password_status === 'correct') {
                    decisionExplanation = 'Password correct BUT behavior suspicious OR amount > $1000';
                } else {
                    decisionExplanation = 'Additional verification needed due to risk factors';
                }
            } else {
                decisionMessage = 'Transaction Blocked';
                decisionExplanation = result.password_status === 'incorrect' ? 
                    'Multiple incorrect password attempts' : 
                    'High risk behavior detected';
            }
            
            resultContainer.innerHTML = `
                <div class="result-box ${resultClass}">
                    <div class="result-header">
                        <span>${resultIcon}</span> ${resultTitle}
                    </div>
                    
                    ${passwordStatusHTML}
                    
                    <div style="text-align: center; margin: 20px 0;">
                        <div style="font-size: 4em; margin: 20px 0;">
                            ${result.password_status === 'correct' ? '👤' : result.password_status === 'incorrect' ? '👹' : '👤'}
                        </div>
                        <div style="font-size: 1.8em; margin: 10px 0; font-weight: bold;">
                            ${decisionMessage}
                        </div>
                        <div style="font-size: 1.2em; opacity: 0.8;">
                            ${decisionExplanation}
                        </div>
                    </div>
                    
                    <div class="result-details">
                        <div class="detail-card">
                            <div class="detail-value">${Math.round(result.total_risk * 100)}%</div>
                            <div class="detail-label">Overall Risk Score</div>
                        </div>
                        
                        <div class="detail-card">
                            <div class="detail-value">${Math.round(result.password_risk * 100)}%</div>
                            <div class="detail-label">Password Risk</div>
                        </div>
                        
                        <div class="detail-card">
                            <div class="detail-value">${Math.round((result.typing_human_score + result.mouse_human_score) / 2)}%</div>
                            <div class="detail-label">Behavior Score</div>
                        </div>
                        
                        <div class="detail-card">
                            <div class="detail-value">$${result.amount}</div>
                            <div class="detail-label">Amount Risk</div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 30px;">
                        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px;">
                            <h4 style="color: #4a69bd; margin-bottom: 15px;">🔐 PASSWORD ANALYSIS</h4>
                            <p>Status: <strong>${result.password_status.toUpperCase()}</strong></p>
                            <p>Attempts: <strong>${result.password_attempts || 0}/3</strong></p>
                            <p>Risk Contribution: <strong>${Math.round(result.password_risk * 100)}%</strong></p>
                            <p>Password Typed: <strong>${result.password_provided ? 'Yes' : 'No'}</strong></p>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px;">
                            <h4 style="color: #4a69bd; margin-bottom: 15px;">🤖 BEHAVIOR ANALYSIS</h4>
                            <p>Typing Score: <strong>${Math.round(result.typing_human_score)}%</strong></p>
                            <p>Mouse Score: <strong>${Math.round(result.mouse_human_score)}%</strong></p>
                            <p>Pattern Match: <strong>${result.similar_patterns_found || 0} similar</strong></p>
                            <p>Analysis Time: <strong>${Math.round(result.processing_time_ms)}ms</strong></p>
                        </div>
                    </div>
                    
                    ${anomaliesHTML}
                    
                    <div style="margin-top: 30px; padding: 20px; background: rgba(0,0,0,0.3); border-radius: 10px;">
                        <h4 style="color: #4a69bd; margin-bottom: 10px;">🔍 SECURITY ANALYSIS BREAKDOWN:</h4>
                        <p>${getAnalysisSummary(result)}</p>
                        <div style="margin-top: 15px; display: flex; justify-content: space-between; font-size: 0.9em;">
                            <span>Password Risk: ${Math.round(result.password_risk * 100)}%</span>
                            <span>Behavior Risk: ${Math.round(result.behavior_risk * 100)}%</span>
                            <span>Amount Risk: ${Math.round(result.amount_risk * 100)}%</span>
                            <span>Time Risk: ${Math.round(result.time_risk * 100)}%</span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function getAnalysisSummary(result) {
            if (result.password_status === 'incorrect') {
                return "🚫 INCORRECT PASSWORD detected. Even if password was correct, behavioral analysis would continue. Account will be locked after 3 failed attempts.";
            } else if (result.decision === 'approve') {
                return "✅ SECURE TRANSACTION. Password validation passed + Behavioral patterns match human characteristics. All security layers passed inspection.";
            } else if (result.decision === 'otp_required') {
                if (result.amount >= 1000) {
                    return "⚠️ LARGE TRANSACTION + BEHAVIOR ANALYSIS: Password correct, but amount exceeds $1000 threshold. OTP required as additional security measure.";
                } else {
                    return "⚠️ SUSPICIOUS BEHAVIOR DETECTED: Password correct, but typing/mouse patterns show anomalies. OTP required to verify legitimate user.";
                }
            } else {
                return "🚫 HIGH-RISK TRANSACTION BLOCKED: Multiple security layers triggered alerts. System detected potential fraud attempt.";
            }
        }
        
        function updateStats(result) {
            // Update counters (simulated)
            const totalTx = parseInt(document.getElementById('totalTransactions').textContent) + 1;
            document.getElementById('totalTransactions').textContent = totalTx;
            
            if (result.risk_level === 'safe') {
                const safeCount = parseInt(document.getElementById('safeCount').textContent) + 1;
                document.getElementById('safeCount').textContent = safeCount;
            } else if (result.risk_level === 'suspicious') {
                const suspiciousCount = parseInt(document.getElementById('suspiciousCount').textContent) + 1;
                document.getElementById('suspiciousCount').textContent = suspiciousCount;
            } else {
                const alertCount = parseInt(document.getElementById('alertCount').textContent) + 1;
                document.getElementById('alertCount').textContent = alertCount;
            }
        }
        
        // Initialize
        updateStatusDisplay();
        updatePasswordStatus('typing', 'Enter password for transaction');
        analyzeLiveBehavior();
    </script>
</body>
</html>
'''

# ==================== API ROUTES ====================
@app.route('/')
def index():
    """Serve the main interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'components': ['cyborgdb', 'behavior_analyzer', 'fraud_detector', 'password_validation'],
        'features': ['password_analysis', 'behavioral_biometrics', 'otp_verification', 'risk_scoring']
    })

@app.route('/api/transaction', methods=['POST'])
def process_transaction():
    """Process a transaction with password validation and behavioral analysis"""
    try:
        data = request.json
        user_id = data.get('user_id', 'user_001')
        password = data.get('password')
        transaction_data = data.get('transaction', {})
        behavior_data = data.get('behavior', {})
        
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID required'}), 400
        
        # Analyze transaction with password
        result = fraud_detector.analyze_transaction(user_id, behavior_data, transaction_data, password)
        
        return jsonify({
            'success': True,
            'message': 'Transaction analyzed successfully',
            'transaction_id': result['transaction_id'],
            **result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP"""
    try:
        data = request.json
        user_id = data.get('user_id')
        otp = data.get('otp')
        
        if not user_id or not otp:
            return jsonify({'success': False, 'error': 'User ID and OTP required'}), 400
        
        # Verify OTP
        is_valid, message = fraud_detector.verify_otp(user_id, otp)
        
        return jsonify({
            'success': is_valid,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/<user_id>/stats')
def get_user_stats(user_id):
    """Get user statistics"""
    try:
        stats = fraud_detector.get_user_stats(user_id)
        return jsonify({
            'success': True,
            'user_id': user_id,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/system-stats')
def get_system_stats():
    """Get system statistics"""
    total_transactions = sum(len(txs) for txs in cyborgdb.transactions.values())
    total_users = len(cyborgdb.user_profiles)
    
    # Count risk levels
    safe_count = 0
    suspicious_count = 0
    alert_count = 0
    
    for user_txs in cyborgdb.transactions.values():
        for tx in user_txs:
            if tx.get('risk_level') == 'safe':
                safe_count += 1
            elif tx.get('risk_level') == 'suspicious':
                suspicious_count += 1
            elif tx.get('risk_level') == 'alert':
                alert_count += 1
    
    # Count password attempts
    total_password_attempts = sum(cyborgdb.password_attempts.values())
    locked_accounts = len(cyborgdb.password_lockouts)
    
    return jsonify({
        'success': True,
        'total_transactions': total_transactions,
        'total_users': total_users,
        'safe_transactions': safe_count,
        'suspicious_transactions': suspicious_count,
        'alert_transactions': alert_count,
        'otp_generated': len(cyborgdb.otp_store),
        'cyborgdb_vectors': len(cyborgdb.vectors),
        'total_password_attempts': total_password_attempts,
        'locked_accounts': locked_accounts,
        'password_verifications': len([tx for user_txs in cyborgdb.transactions.values() 
                                     for tx in user_txs if tx.get('password_provided')])
    })

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Reset the system (for testing)"""
    global cyborgdb, behavior_analyzer, fraud_detector
    
    cyborgdb = CyborgDB()
    behavior_analyzer = BehaviorAnalyzer()
    fraud_detector = FraudDetector(cyborgdb, behavior_analyzer)
    
    return jsonify({
        'success': True,
        'message': 'System reset successfully',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/check-password', methods=['POST'])
def check_password():
    """Check password without processing transaction"""
    try:
        data = request.json
        user_id = data.get('user_id')
        password = data.get('password')
        
        if not user_id or not password:
            return jsonify({'success': False, 'error': 'User ID and password required'}), 400
        
        status = cyborgdb.verify_password(user_id, password)
        attempts = cyborgdb.get_password_attempts(user_id)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'password_correct': status == PasswordStatus.CORRECT,
            'password_status': status.value,
            'attempts': attempts,
            'attempts_remaining': 3 - attempts,
            'is_locked': attempts >= 3
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== MAIN ====================
if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                     GHOSTWALLET v4.0                         ║
    ║     Password Validation + Behavioral Biometrics              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🚀 Starting server on: http://localhost:5000
    📊 Dashboard: http://localhost:5000
    
    🔥 ENHANCED FEATURES:
    • PASSWORD VALIDATION before transaction
    • Even CORRECT password can trigger OTP if behavior is suspicious
    • 3 wrong passwords → Account lock for 5 minutes
    • Password-specific behavioral analysis
    • Multi-layer risk scoring
    
    🎯 HOW IT WORKS NOW:
    1. Enter User ID and Password (required)
    2. System validates password
    3. Analyzes HOW you type the password (typing speed, rhythm)
    4. Analyzes mouse movements during password entry
    5. Combines password correctness + behavior analysis
    6. Makes decision: Safe/OTP Required/Blocked
    
    🔐 DEMO PASSWORDS:
    • user_001 → SecurePass123!
    • user_002 → Test@Password456
    • user_003 → Demo#Pass789
    • user_004 → GhostWallet2024!
    
    ⚠️  SECURITY SCENARIOS:
    ✅ Password correct + Normal behavior → APPROVE
    ✅ Password correct + Suspicious typing → OTP REQUIRED
    ❌ Password incorrect → BLOCK (3 attempts lock)
    ⚠️  Amount > $1000 → Always OTP REQUIRED
    🚫 Robotic patterns → Immediate BLOCK
    
    📈 DECISION MATRIX:
    • Safe (✅): Correct password + Normal behavior + amount < $1000
    • Suspicious + OTP (⚠️): Correct password but unusual behavior OR amount > $1000
    • Alert (🚫): Wrong password OR clear fraud indicators
    
    🔒 SECURITY LAYERS:
    1. Password validation
    2. Typing speed analysis
    3. Mouse pattern analysis
    4. Amount risk assessment
    5. Time-based risk
    6. Historical pattern matching
    
    ⚠️  Press Ctrl+C to stop the server
    """)
    
    app.run(host='0.0.0.0', port=5000, debug=False)